"""
结尾质量分析 —— 量化"开始/结束"表达能力。

参照基准 (人类语料, 按 _call 键检测后判断末音是否落在本曲主音):
    - 末音 == 本曲主音 的比例
    - 末音为长音 (>= 附点四分) 的比例
    - 末 4 音包含 V→I / vii°→I 类解决的比例

生成曲用同一套口径统计。用法:
    python analyze_endings.py [--n 8] [--pt melody_diffusion_v6_struct.pt]
"""
import sys, os, json, argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import torch
from metrics.theory_metrics import load_midi_notes, evaluate_key
from constants import MELODY_RHYTHM


def corpus_reference():
    d = json.load(open(ROOT / 'data/processed/melody_llm_full_v2.json', encoding='utf-8'))
    pieces, cur = [], []
    for n in d:
        if n.get('prev_pc') is None and cur:
            pieces.append(cur); cur = []
        cur.append(n)
    if cur:
        pieces.append(cur)
    tonic_end = long_end = cad_like = total = 0
    for p in pieces:
        pcs = [n['pc'] for n in p if n['pc'] < 12]
        if len(pcs) < 6:
            continue
        total += 1
        # 调性检测 (用整曲音级直方图)
        from metrics.theory_metrics import key_confidence
        import numpy as np
        hist = np.zeros(12)
        for pc in pcs:
            hist[pc % 12] += 1
        k = key_confidence(hist)
        tonic_pc = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}[k['key'][:2].strip()]
        last = pcs[-1] % 12
        if last == tonic_pc:
            tonic_end += 1
        # 末音时值
        last_rh = [n['rhythm'] for n in p if n['pc'] < 12][-1]
        if last_rh >= 5:
            long_end += 1
        # 末 3 音是否含 导音/上主音 → 主音 的级进解决
        tail = [pc % 12 for pc in pcs[-3:]]
        if len(tail) >= 2:
            d_ = (tail[-2] - tail[-1]) % 12
            if d_ in (11, 2) or (tail[-2] - tail[-1]) % 12 == 10:  # 7→0, 2→0, 10→0(下行)
                cad_like += 1
    return {'n': total, 'tonic_end': tonic_end / total, 'long_end': long_end / total,
            'cad_like': cad_like / total}


def gen_reference(midi_paths):
    tonic_end = long_end = cad_like = total = target_tonic = 0
    keys = {}
    for p in midi_paths:
        notes = load_midi_notes(str(p), track=1)
        if len(notes) < 6:
            continue
        total += 1
        k = evaluate_key(notes)
        tonic_pc = {'C': 0, 'C#': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5,
                    'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}[k['key'][:2].strip()]
        pcs = [n.pitch % 12 for n in notes]
        if pcs[-1] == tonic_pc:
            tonic_end += 1
        if pcs[-1] == 0:                     # 目标主音 C (生成限定 C 大调)
            target_tonic += 1
        keys[k['key']] = keys.get(k['key'], 0) + 1
        if notes[-1].dur_beats >= 1.5:
            long_end += 1
        if len(pcs) >= 2:
            d_ = (pcs[-2] - pcs[-1]) % 12
            if d_ in (11, 2, 10):
                cad_like += 1
    return {'n': total, 'tonic_end': tonic_end / max(total, 1),
            'long_end': long_end / max(total, 1),
            'cad_like': cad_like / max(total, 1),
            'target_tonic': target_tonic / max(total, 1),
            'keys': keys}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--bpm', type=int, default=92)
    parser.add_argument('--midi', type=str, nargs='*', default=None,
                        help='直接评估给定 MIDI (渲染后口径); 缺省时用 gen_standalone 现场生成')
    args = parser.parse_args()

    ref = corpus_reference()
    print(f"人类语料参照 (n={ref['n']}): 末音=主音 {ref['tonic_end']*100:.0f}% | "
          f"末音长音 {ref['long_end']*100:.0f}% | 末3音含解决 {ref['cad_like']*100:.0f}%")

    if args.midi:
        paths = [Path(p) for p in args.midi]
        paths = [p if p.is_absolute() else ROOT / p for p in paths]
    else:
        # 生成 n 首 (独立生成器, 与管线同后处理)
        import subprocess
        paths = []
        for i in range(args.n):
            out = ROOT / f'data/generated/_endtest_{i}.mid'
            subprocess.run([sys.executable, str(ROOT / 'gen_standalone.py'),
                            '--seed', str(100 + i), '--chords', '24',
                            '--bpm', str(args.bpm), '--out', str(out)],
                           capture_output=True, text=True)
            if out.exists():
                paths.append(out)
    gen = gen_reference(paths)
    print(f"生成曲 (n={gen['n']}):    末音=检测主音 {gen['tonic_end']*100:.0f}% | "
          f"末音=C(目标) {gen['target_tonic']*100:.0f}% | "
          f"末音长音 {gen['long_end']*100:.0f}% | 末3音含解决 {gen['cad_like']*100:.0f}%")
    print(f"  检测调性分布: {gen['keys']}")
