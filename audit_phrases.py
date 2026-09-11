"""
乐句标注质量审计 —— 规则标注 (终止式+长音吸附) vs 本地 LLM 分段。

对抽样的众赞歌: 把旋律 (音名/时值/小节线) 给 qwen3.5:9b (think=false),
让它标出乐句末; 与规则标注对比 (允许 ±1 音容差), 报告一致率。
用途: 判断"句子"标签的可信度 —— 这是模型能否学会句法的前提。

用法: python audit_phrases.py [--n 40]
"""
import sys, os, json, random, argparse, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))

import urllib.request
from gen_sentence import piece_conditions

OLLAMA = 'http://localhost:11434/api/chat'
PN = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

SYSTEM = """You are a music analyst. Given a chorale melody, mark where each phrase ends
(the last note of each phrase = where a singer breathes). Chorale phrases end at
cadences. Reply ONLY with JSON: {"ends": [note indices]}."""


def call(prompt, timeout=180):
    body = {'model': 'qwen3.5:9b-q4_K_M',
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': prompt}],
            'stream': False, 'think': False,
            'options': {'temperature': 0.2, 'num_predict': 256}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        txt = json.loads(resp.read().decode())['message']['content']
    import re
    m = re.search(r'\{.*\}', txt, re.S)
    return json.loads(m.group(0)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--out', type=str, default='data/processed/phrase_audit.json')
    args = parser.parse_args()

    data = json.load(open(ROOT / 'data/processed/chorales_sentences_v1.json', encoding='utf-8'))
    pool = [p for p in data if len(p['notes']) >= 24]
    rng = random.Random(3)
    sample = rng.sample(pool, min(args.n, len(pool)))

    agree, total_rule, total_llm, matched = 0, 0, 0, 0
    rows = []
    t0 = time.time()
    for k, p in enumerate(sample):
        notes = p['notes']
        n = len(notes)
        # 规则标注 (排除曲末)
        cond = piece_conditions(p)
        rule_ends = sorted(i for i in range(n - 1) if cond['phrase_bin'][i] == 0)
        # LLM 分段
        toks, t = [], 0.0
        for i, nt in enumerate(notes):
            bar = ' |' if i > 0 and int(t) % 4 == 0 else ''
            toks.append(f"{bar} {i}:{PN[nt['pc']]}:{nt['ql']:g}".strip())
            t += nt['ql']
        prompt = (f"Chorale melody ({n} notes, C is tonic):\n" + ' '.join(toks) +
                  f"\nMark the last note of each phrase (breath points). "
                  f"Reply ONLY with JSON: {{\"ends\": [indices]}}")
        try:
            res = call(prompt)
            llm_ends = sorted({int(x) for x in res.get('ends', [])
                               if isinstance(x, (int, float)) and 0 <= int(x) < n})
        except Exception:
            llm_ends = []
        if llm_ends:
            # 容差 ±1 匹配
            m_rule = sum(1 for e in rule_ends if any(abs(e - l) <= 1 for l in llm_ends))
            m_llm = sum(1 for l in llm_ends if any(abs(l - e) <= 1 for e in rule_ends))
            total_rule += len(rule_ends); total_llm += len(llm_ends)
            matched += m_rule
            rows.append({'id': p['id'], 'title': p['title'], 'n': n,
                         'rule': rule_ends, 'llm': llm_ends,
                         'prec': m_llm / len(llm_ends), 'rec': m_rule / len(rule_ends)})
            agree += 1 if (m_rule == len(rule_ends) and m_llm == len(llm_ends)) else 0
        if (k + 1) % 10 == 0:
            print(f'  {k+1}/{len(sample)} | 已用 {time.time()-t0:.0f}s')

    report = {'n_audited': len(rows),
              'rule_recall_vs_llm': matched / max(total_rule, 1),
              'llm_precision_vs_rule': sum(r['prec'] * len(r['llm']) for r in rows) / max(total_llm, 1),
              'exact_agreement': agree / max(len(rows), 1)}
    print(f"审计完成: {report['n_audited']} 首")
    print(f"  规则标注被 LLM 认可 (召回): {report['rule_recall_vs_llm']*100:.0f}%")
    print(f"  LLM 标注落在规则标注附近 (精确): {report['llm_precision_vs_rule']*100:.0f}%")
    print(f"  完全一致: {report['exact_agreement']*100:.0f}%")
    json.dump({'summary': report, 'rows': rows},
              open(ROOT / args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
