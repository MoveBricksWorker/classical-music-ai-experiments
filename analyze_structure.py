"""
符号域结构分析 (SSM) —— 参考 Toward Guided Musical Form (CMJ 2024) 的
自相似矩阵/FSD 思路, 适配到符号旋律。

对每首曲子:
    1. 按拍切分, 每个时间片提取 [12维音级直方图 + 节奏密度] 特征向量;
    2. 计算自相似矩阵 SSM (余弦相似度);
    3. 提取结构特征:
       - rep_ratio   : 非对角区域相似度 > 阈值 的比例 (重复材料占比)
       - block_score : 对角外平行块结构的强度 (对角线平行带均值)
       - novelty_rate: 相邻时间片相似度的下降率 (新材料出现速度)
    4. 生成集 vs 人类语料的特征分布对比 (FSD 精神的简化版)。

用法:
    python analyze_structure.py --gen-n 8
"""
import sys, json, argparse, glob
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np

from metrics.theory_metrics import load_midi_notes
from constants import MELODY_RHYTHM


def note_features(notes, beat_step: float = 0.5) -> np.ndarray:
    """按拍切片 → 每片 [12 维音级直方图 (按时值加权) + 节奏密度]。"""
    if not notes:
        return np.zeros((0, 13))
    total_beats = max(n.start_beat + n.dur_beats for n in notes)
    n_slices = max(int(np.ceil(total_beats / beat_step)), 1)
    feats = np.zeros((n_slices, 13))
    for n in notes:
        i = int(n.start_beat / beat_step)
        if i >= n_slices:
            continue
        feats[i, n.pitch % 12] += n.dur_beats
        feats[i, 12] += 1
    # 平滑 (3 片窗口), 归一化
    k = np.array([0.25, 0.5, 0.25])
    for j in range(feats.shape[1]):
        feats[:, j] = np.convolve(feats[:, j], k, mode='same')
    norm = np.linalg.norm(feats, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return feats / norm


def ssm(feats: np.ndarray) -> np.ndarray:
    if feats.shape[0] == 0:
        return np.zeros((0, 0))
    return feats @ feats.T


def structural_features(notes, beat_step: float = 0.5) -> dict:
    """重复比例 / 平行块强度 / 新颖率。"""
    feats = note_features(notes, beat_step)
    if feats.shape[0] < 8:
        return {'rep_ratio': 0.0, 'block_score': 0.0, 'novelty_rate': 0.0, 'n_slices': feats.shape[0]}
    S = ssm(feats)
    n = S.shape[0]
    # 去掉近对角带 (自相似噪声), 只看 |i-j| >= 4 的区域
    off = 4
    mask = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= off
    vals = S[mask]
    rep_ratio = float((vals > 0.85).mean())          # 高相似重复比例
    # 平行块强度: 固定 lag 的相似度均值 (lag 4..n//2), 取最大几个
    lags = range(off, n // 2 + 1)
    lag_means = [float(np.mean(np.diag(S, k))) for k in lags] if n > 2 * off else [0.0]
    block_score = float(np.mean(sorted(lag_means, reverse=True)[:3])) if lag_means else 0.0
    # 新颖率: 相邻片相似度下降到低值的比例
    adj = np.diag(S, 1)
    novelty_rate = float((adj < 0.5).mean()) if len(adj) else 0.0
    return {'rep_ratio': rep_ratio, 'block_score': block_score,
            'novelty_rate': novelty_rate, 'n_slices': n}


def corpus_features(max_pieces: int = 120):
    d = json.load(open(ROOT / 'data/processed/melody_llm_full_v2.json', encoding='utf-8'))
    pieces, cur = [], []
    for n in d:
        if n.get('prev_pc') is None and cur:
            pieces.append(cur); cur = []
        cur.append(n)
    if cur:
        pieces.append(cur)
    out = []
    for p in pieces[:max_pieces]:
        notes = []
        t = 0.0
        for n in p:
            dur = MELODY_RHYTHM.get(n['rhythm'], 0.5)
            out_n = type('N', (), {'start_beat': t, 'dur_beats': dur,
                                   'pitch': (n['pc'] % 12) + 60})()
            if n['pc'] < 12:
                notes.append(out_n)
            t += dur
        out.append(structural_features(notes))
    return out


def gen_features(paths):
    out = []
    for p in paths:
        notes = load_midi_notes(str(p), track=1)
        out.append(structural_features(notes))
    return out


def summarize(feats, label):
    keys = ['rep_ratio', 'block_score', 'novelty_rate']
    stats = {k: (float(np.mean([f[k] for f in feats])), float(np.std([f[k] for f in feats])))
             for k in keys}
    print(f'{label} (n={len(feats)}):')
    for k in keys:
        m, s = stats[k]
        print(f'   {k:14s}: {m:.3f} ± {s:.3f}')
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-n', type=int, default=8)
    args = parser.parse_args()

    print('=== 符号域 SSM 结构分析 (Guided Musical Form 思路) ===')
    ref = corpus_features()
    summarize(ref, '人类语料')

    # 生成曲: 复用 endtest (若无则现生成)
    paths = sorted(glob.glob(str(ROOT / 'data/generated/_endtest_*.mid')))
    if len(paths) < args.gen_n:
        import subprocess
        paths = []
        for i in range(args.gen_n):
            out = ROOT / f'data/generated/_struct_{i}.mid'
            subprocess.run([sys.executable, str(ROOT / 'gen_standalone.py'),
                            '--seed', str(200 + i), '--chords', '24', '--bpm', '92',
                            '--out', str(out)], capture_output=True, text=True)
            if out.exists():
                paths.append(out)
    gen = gen_features(paths[:args.gen_n])
    summarize(gen, '生成曲 (v7 管线)')

    # FSD 精神的简化: 特征分布的标准化距离
    print()
    for k in ['rep_ratio', 'block_score', 'novelty_rate']:
        ref_m = np.mean([f[k] for f in ref]); ref_s = np.std([f[k] for f in ref]) + 1e-6
        gen_m = np.mean([f[k] for f in gen])
        print(f'   {k:14s}: 生成-人类 标准化距离 = {abs(gen_m - ref_m) / ref_s:.2f} σ')
