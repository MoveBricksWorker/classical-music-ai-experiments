"""
乐句边界标注 —— 用本地 Ollama 模型给旋律语料标注"换气点"。

任务: 每首曲子是若干乐句 ("句子") 组成, 歌手需要在乐句边界换气。
给模型音符序列 (音名+时值+小节线), 让它标出每个乐句的**最后一个音**的序号。

输出: data/processed/melody_phrases_v1.json
    {"pieces": [{"start": 全局起始音符索引, "len": 长度, "breaths": [句子末尾局部序号...]}, ...]}

特性: 批量请求 / 断点续传 / JSON 校验 + 单曲重试 / 规则兜底 (休止与长音)。
用法:
    python annotate_phrases.py [--model qwen3.5:9b-q4_K_M] [--batch 4] [--limit N]
"""
import sys, os, json, time, argparse, re
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

import urllib.request

from constants import MELODY_RHYTHM

OLLAMA = 'http://localhost:11434/api/chat'
PITCH_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

SYSTEM = """You are a music analyst. You segment monophonic melodies into musical phrases.
A phrase is a musical "sentence" a singer can sing in one breath. A breath (phrase boundary)
typically happens after a longer note, a rest, or a cadential gesture (e.g. leading tone to tonic).
Phrases are usually 4-12 notes long. The LAST note of each phrase is the breath point.
Always include the final note of the piece as a breath point."""


def fmt_dur(rh: int) -> str:
    d = MELODY_RHYTHM.get(rh, 0.5)
    return {0.125: '32nd', 0.25: '16th', 0.5: '8th', 0.75: 'd8th',
            1.0: 'q', 1.5: 'dq', 2.0: 'h', 4.0: 'w'}.get(d, str(d))


def format_piece(notes) -> str:
    """音名+时值+小节线 (拍位由累计时值推算)。"""
    tokens = []
    t = 0.0
    for i, n in enumerate(notes):
        bar = ''
        if i == 0 or int(t + 1e-6) // 4 != int(t - MELODY_RHYTHM.get(n['rhythm'], 0.5) + 1e-6) // 4:
            if i > 0:
                bar = ' |'
        if n['pc'] >= 12:
            tok = 'REST'
        else:
            tok = PITCH_NAMES[n['pc'] % 12]
        tokens.append(f"{bar} {i}:{tok}:{fmt_dur(n['rhythm'])}".strip())
        t += MELODY_RHYTHM.get(n['rhythm'], 0.5)
    return ' '.join(tokens)


def call_ollama(model, prompt, timeout=300):
    body = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': prompt},
        ],
        'stream': False,
        'think': False,          # 关闭思考: qwen3.5 思考链极慢且占满 num_predict
        'options': {'temperature': 0.2, 'num_predict': 512},
    }
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data['message']['content']


def parse_json(text):
    """从回复中提取 JSON (容忍 markdown 代码块与前后缀)。"""
    text = text.strip()
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def validate(breaths, L: int):
    """校验并清洗边界: 升序去重, 间隔 >= 2, 含末位。"""
    if not isinstance(breaths, list):
        return None
    out = sorted({int(b) for b in breaths if isinstance(b, (int, float)) and 0 <= int(b) < L})
    if not out:
        return None
    cleaned = []
    for b in out:
        if not cleaned or b - cleaned[-1] >= 2:
            cleaned.append(b)
    if cleaned[-1] != L - 1:
        cleaned.append(L - 1)
    if len(cleaned) < 2:
        return None            # 全局一个乐句 → 视为无效, 交兜底
    return cleaned


def fallback_breaths(notes):
    """规则兜底: 休止后 / 长音 (>=2拍) 处断句。"""
    L = len(notes)
    out = []
    t = 0.0
    for i, n in enumerate(notes):
        dur = MELODY_RHYTHM.get(n['rhythm'], 0.5)
        if i > 0 and i < L - 1:
            prev_rest = notes[i - 1]['pc'] >= 12
            if prev_rest or dur >= 4.0:
                out.append(i - 1)
        t += dur
    out.append(L - 1)
    cleaned = []
    for b in out:
        if not cleaned or b - cleaned[-1] >= 2:
            cleaned.append(b)
    return cleaned if len(cleaned) >= 2 else [L - 1]


def build_prompt(batch):
    """batch: [(piece_id, notes)]"""
    parts = []
    for pid, notes in batch:
        parts.append(f'Piece {pid} ({len(notes)} notes):\n{format_piece(notes)}')
    ids = ', '.join(f'"{pid}"' for pid, _ in batch)
    parts.append(
        f'\nMark the breath points (index of the last note of each phrase) for each piece.\n'
        f'Reply ONLY with JSON: {{"<piece id>": [breath indices], ...}} for ids: {ids}.')
    return '\n\n'.join(parts)


def load_pieces(json_path):
    d = json.load(open(json_path, encoding='utf-8'))
    pieces, cur, start = [], [], 0
    for i, n in enumerate(d):
        if n.get('prev_pc') is None and cur:
            pieces.append({'start': start, 'len': len(cur), 'notes': cur})
            cur, start = [], i
        cur.append(n)
    if cur:
        pieces.append({'start': start, 'len': len(cur), 'notes': cur})
    return d, pieces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3.5:9b-q4_K_M')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='只处理前 N 首 (调试)')
    parser.add_argument('--out', type=str, default='data/processed/melody_phrases_v1.json')
    args = parser.parse_args()

    source = ROOT / 'data/processed/melody_llm_full_v2.json'
    out_path = ROOT / args.out
    _, pieces = load_pieces(source)
    print(f'曲目数: {len(pieces)}, 总音符: {sum(p["len"] for p in pieces)}')

    # 断点续传
    done = {}
    if out_path.exists():
        prev = json.load(open(out_path, encoding='utf-8'))
        done = {p['start']: p for p in prev['pieces']}
        print(f'已标注 {len(done)} 首, 续跑')

    todo = [p for p in pieces if p['start'] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'待标注: {len(todo)} 首, 模型: {args.model}, 批量: {args.batch}')

    t0 = time.time()
    n_llm = n_fallback = n_fail = 0
    for bi in range(0, len(todo), args.batch):
        batch = [(p['start'], p['notes']) for p in todo[bi:bi + args.batch]]
        prompt = build_prompt(batch)
        result = None
        for attempt in range(3):
            try:
                txt = call_ollama(args.model, prompt)
                result = parse_json(txt)
                if result:
                    break
            except Exception as e:
                print(f'  请求失败 (尝试 {attempt+1}): {e}')
                time.sleep(2)
        for pid, notes in batch:
            L = len(notes)
            breaths = None
            if result and str(pid) in result:
                breaths = validate(result[str(pid)], L)
            if breaths:
                n_llm += 1
                src = 'llm'
            else:
                breaths = fallback_breaths(notes)
                n_fallback += 1
                src = 'fallback'
            done[pid] = {'start': pid, 'len': L, 'breaths': breaths, 'source': src}
        # 增量落盘
        json.dump({'pieces': [done[k] for k in sorted(done)]},
                  open(out_path, 'w', encoding='utf-8'), ensure_ascii=False)
        if (bi // args.batch) % 10 == 0:
            el = time.time() - t0
            print(f'  {len(done)}/{len(pieces)} 首 | LLM {n_llm} 兜底 {n_fallback} | {el:.0f}s')

    print(f'完成: {len(done)} 首 (LLM {n_llm} / 兜底 {n_fallback}) → {out_path}')


if __name__ == '__main__':
    main()
