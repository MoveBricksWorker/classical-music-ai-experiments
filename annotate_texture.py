"""
织体标注审计 —— 用本地 Ollama 模型独立复核规则标注的织体类型。

规则标注（`build_chorale_satb_dataset.py`）可复现、覆盖全量，但阈值是人工定的。
按项目惯例（见 00-交接文档 §6："任何标注都先审计再信"），这里抽一批乐句，
把四声部切片的可读表格交给本地模型独立判定，报告**一致率 / 混淆 / 分歧案例**。

注意口径：LLM 从文本表格判断对位与模仿本来就弱，因此本脚本的定位是
**审计规则**（找规则的明显错判）而不是充当真值。一致率低只说明需要人工抽查，
不能反过来说规则错。

输出: data/processed/texture_audit.json
    {"summary": {"n", "agree", "agreement", "confusion": {...}, "by_label": {...}},
     "rows": [{"id", "phrase", "rule", "llm", "agree", "reason"}]}

用法:
    python annotate_texture.py [--n 60] [--model qwen3.5:9b-q4_K_M] [--batch 3] [--seed 0]
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

OLLAMA = 'http://localhost:11434/api/chat'
VOICES = ['S', 'A', 'T', 'B']
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
LABELS = ['chordal', 'polyphonic', 'imitative', 'pedal', 'unison_octave', 'mixed']

SYSTEM = """You are a music theory analyst specializing in Baroque vocal polyphony (Bach chorales).
You are given a short four-voice passage (S/A/T/B) as a table of vertical slices.
Each row is one time point; each cell is the pitch sounding in that voice:
  "C5"   = a new note attacked here
  "C5~"  = the same note held over from the previous row
  "-"    = the voice is silent
Classify the passage's TEXTURE into exactly ONE of these labels:
  chordal        - voices move together in the same rhythm (homophonic / block chords)
  polyphonic     - voices have independent rhythms, no clear motivic imitation
  imitative      - a short motif is passed from one voice to another (point of imitation)
  pedal          - one voice (usually the bass) holds a long note while the others move above it
  unison_octave  - the voices move together in unison/octaves (monophonic in effect)
  mixed          - no single type dominates
Base your judgement on the rhythm relationships between voices, the bass behaviour, and whether
voices share the same melodic material. Reply with JSON only."""


def midi_name(m: int) -> str:
    return f'{NAMES[m % 12]}{m // 12 - 1}'


def fmt_phrase(piece, ph, idx) -> str:
    seg = piece['slices'][ph['start_slice']:ph['end_slice'] + 1]
    rows = []
    for s in seg:
        cells = []
        for v in VOICES:
            m = s['midi'][v]
            if m is None:
                cells.append('-')
            else:
                cells.append(midi_name(m) + ('' if s['attack'][v] else '~'))
        rows.append(f"{s['off']:>6.2f} | " + ' | '.join(f'{c:>5}' for c in cells))
    head = (f'Piece: {piece["title"][:50]!r} (key {piece["key_original"]}, '
            f'{piece["n_measures"]} bars, 4/4)\n'
            f'Phrase {idx + 1} of {len(piece["phrases"])}, '
            f'cadence={ph["cadence"]}, {len(seg)} slices')
    return head + '\n' + '\n'.join(rows)


def call_ollama(model, prompt, timeout=300):
    body = {
        'model': model,
        'messages': [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': prompt}],
        'stream': False,
        'think': False,                     # 思考型模型必须关闭, 否则极慢 (见 00 文档)
        'options': {'temperature': 0.0, 'num_predict': 512},
    }
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())['message']['content']


def parse_json(text):
    m = re.search(r'\{.*\}', text.strip(), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_prompt(batch):
    parts = []
    for pid, text in batch:
        parts.append(f'--- ITEM {pid} ---\n{text}')
    ids = ', '.join(f'"{pid}"' for pid, _ in batch)
    parts.append(f'\nClassify each item. Reply ONLY with JSON: '
                 f'{{"<item id>": {{"texture": "<label>", "confidence": 0.0, '
                 f'"reason": "<one short sentence>"}}, ...}} for ids: {ids}.')
    return '\n\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='data/processed/chorales_satb_v2.json')
    ap.add_argument('--n', type=int, default=60, help='抽样乐句数')
    ap.add_argument('--model', type=str, default='qwen3.5:9b-q4_K_M')
    ap.add_argument('--batch', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='data/processed/texture_audit.json')
    args = ap.parse_args()

    pieces = json.load(open(ROOT / args.data, encoding='utf-8'))
    rng = random.Random(args.seed)
    items = []                              # (id, piece, phrase, phrase_idx)
    for p in pieces:
        for i, ph in enumerate(p['phrases']):
            if ph['texture'] in (None, 'unknown'):
                continue
            items.append((f'{p["id"]}#{i}', p, ph, i))
    # 分层抽样: 按规则标签分层, 保证每类都抽到
    by_label = {}
    for it in items:
        by_label.setdefault(it[2]['texture'], []).append(it)
    per = max(1, args.n // max(len(by_label), 1))
    sample = []
    for lab, lst in sorted(by_label.items()):
        rng.shuffle(lst)
        sample += lst[:per]
    rng.shuffle(sample)
    print(f'候选乐句 {len(items)} | 规则标签分布 '
          f'{ {k: len(v) for k, v in sorted(by_label.items())} }')
    print(f'分层抽样 {len(sample)} 条, 模型 {args.model}, 批量 {args.batch}')

    rows = []
    t0 = time.time()
    for bi in range(0, len(sample), args.batch):
        batch = [(it[0], fmt_phrase(it[1], it[2], it[3])) for it in sample[bi:bi + args.batch]]
        res = None
        for attempt in range(3):
            try:
                res = parse_json(call_ollama(args.model, build_prompt(batch)))
                if res:
                    break
            except Exception as e:
                print(f'  请求失败 ({attempt+1}/3): {e}')
                time.sleep(2)
        for pid, _ in batch:
            it = next(x for x in sample[bi:bi + args.batch] if x[0] == pid)
            rule = it[2]['texture']
            got = (res or {}).get(str(pid)) or (res or {}).get(pid) or {}
            llm = str(got.get('texture', '')).strip().lower().replace(' ', '_')
            if llm not in LABELS:
                llm = 'invalid'
            rows.append({'id': pid, 'rule': rule, 'llm': llm,
                         'agree': llm == rule,
                         'confidence': got.get('confidence'),
                         'reason': (got.get('reason') or '')[:200]})
        print(f'  {min(bi + args.batch, len(sample))}/{len(sample)} | '
              f'已用时 {time.time()-t0:.0f}s', flush=True)

    valid = [r for r in rows if r['llm'] != 'invalid']
    agree = sum(1 for r in valid if r['agree'])
    confusion = {}
    for r in valid:
        confusion.setdefault(r['rule'], {}).setdefault(r['llm'], 0)
        confusion[r['rule']][r['llm']] += 1
    by_label = {}
    for r in valid:
        d = by_label.setdefault(r['rule'], {'n': 0, 'agree': 0})
        d['n'] += 1
        d['agree'] += int(r['agree'])
    summary = {
        'n_sampled': len(rows), 'n_valid': len(valid),
        'n_invalid': len(rows) - len(valid),
        'agreement': round(agree / max(len(valid), 1), 3),
        'by_label': {k: {**v, 'agreement': round(v['agree'] / max(v['n'], 1), 3)}
                     for k, v in by_label.items()},
        'confusion': confusion,
        'model': args.model,
    }
    json.dump({'summary': summary, 'rows': rows},
              open(ROOT / args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n=== 审计结果 ===')
    print(f"  有效样本 {len(valid)}/{len(rows)} | 一致率 {summary['agreement']:.0%}")
    for lab, d in sorted(summary['by_label'].items()):
        print(f"    {lab:14s} n={d['n']:3d} 一致 {d['agreement']:.0%}")
    print(f"  分歧案例已存 → {args.out}")

    dis = [r for r in valid if not r['agree']][:6]
    if dis:
        print('\n  典型分歧 (规则 → LLM):')
        for r in dis:
            print(f"    {r['id']:22s} {r['rule']:14s} → {r['llm']:14s} | {r['reason'][:80]}")


if __name__ == '__main__':
    main()
