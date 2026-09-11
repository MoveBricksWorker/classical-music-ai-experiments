"""
防"统计模仿"证据指标 —— 参考 MusicLDM (#11) 与 Deep Recurrent Music Writer (#2)。

    simaa       : SIMAA@90/95 —— 生成样本与训练集最大相似度 ≥ 阈值的比例
                  (MusicLDM 口径: BLM 使 0.047 → 0.020)
    mahalanobis : 17 维符号特征 + 马氏距离 + 阈值标定 + Welch t 检验
                  (DRMW 口径: ≤6 连贯 / 3-4 接近风格 / ≥10 随机)

相似度定义: 音高三元组 (pitch-class trigram) Jaccard —— 符号域对 MusicLDM
音频嵌入相似度的直接映射, 无需额外模型。

17 维特征 (透明可解释):
    12 维 pitch-class 直方图 (归一化, 休止不计数)
    + 平均时值 (beat) + 每拍音符密度 + 平均音程 (半音)
    + step ratio (≤2) + leap ratio (≥7)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from itertools import islice

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from constants import MELODY_RHYTHM  # noqa: E402
from metrics.theory_metrics import REST_PC  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 序列工具
# ─────────────────────────────────────────────────────────────────────────────

def _trigrams(tokens: list[int]) -> set[tuple[int, ...]]:
    return set(zip(tokens, tokens[1:], tokens[2:]))


def trigram_jaccard(a: list[int], b: list[int]) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def windows(tokens: list[int], seq: int = 16, stride: int = 4) -> list[list[int]]:
    """与训练一致的滑窗切分。"""
    return [tokens[i:i + seq] for i in range(0, len(tokens) - seq + 1, stride)
            if len(tokens[i:i + seq]) == seq]


# ─────────────────────────────────────────────────────────────────────────────
# SIMAA
# ─────────────────────────────────────────────────────────────────────────────

def simaa(train_tokens: list[int], gen_sequences: list[list[int]],
          train_seq: int = 16, train_stride: int = 4,
          thresholds=(0.90, 0.95)) -> dict:
    """SIMAA: 生成序列对训练窗口的最大三元组 Jaccard 相似度。

    train_tokens : 完整训练 token 流 (内部按 train_seq/stride 切窗)
    gen_sequences: 生成序列列表 (每条 ≤ train_seq 长)
    """
    train_win = windows(train_tokens, train_seq, train_stride)
    train_sets = [_trigrams(w) for w in train_win]

    def max_sim(gen: list[int]) -> float:
        tg = _trigrams(gen)
        if not tg:
            return 0.0
        best = 0.0
        for ts in train_sets:
            inter = len(tg & ts)
            if inter:
                j = inter / len(tg | ts)
                if j > best:
                    best = j
        return best

    sims = [max_sim(g) for g in gen_sequences]
    sims = np.array(sims)

    # 记忆率: 生成三元组中出现在训练集中的比例
    train_trigram_pool = set().union(*train_sets) if train_sets else set()
    memory = []
    for g in gen_sequences:
        tg = _trigrams(g)
        memory.append(len(tg & train_trigram_pool) / len(tg) if tg else 1.0)

    out = {
        'n_generated': len(sims),
        'mean_max_sim': float(sims.mean()),
        'median_max_sim': float(np.median(sims)),
        'max_max_sim': float(sims.max()),
        'mean_trigram_memory': float(np.mean(memory)),
    }
    for t in thresholds:
        out[f'SIMAA@{int(t * 100)}'] = float((sims >= t).mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 17 维特征 + 马氏距离
# ─────────────────────────────────────────────────────────────────────────────

def melody_features(pitches: list[int], rhythms: list[int]) -> np.ndarray:
    """17 维符号特征。"""
    f = np.zeros(17)
    valid = [(p, r) for p, r in zip(pitches, rhythms) if p < REST_PC]
    if not valid:
        return f
    pcs = [p for p, _ in valid]
    durs = [MELODY_RHYTHM.get(r, 0.5) for _, r in valid]

    hist = np.zeros(12)
    for p in pcs:
        hist[p % 12] += 1
    f[:12] = hist / hist.sum()

    total_beats = sum(durs)
    intervals = []
    for a, b in zip(pcs[:-1], pcs[1:]):
        d = abs(a - b)
        intervals.append(min(d, 12 - d))
    arr = np.array(intervals) if intervals else np.zeros(0)

    f[12] = float(np.mean(durs))                       # 平均时值
    f[13] = len(valid) / total_beats if total_beats > 0 else 0.0  # 每拍密度
    f[14] = float(arr.mean()) if len(arr) else 0.0     # 平均音程
    f[15] = float((arr <= 2).mean()) if len(arr) else 0.0  # step ratio
    f[16] = float((arr >= 5).mean()) if len(arr) else 0.0  # leap ratio (>=四度)
    return f


@dataclass
class MahalanobisReport:
    train_mean: float
    train_std: float
    gen_mean: float
    gen_std: float
    t_stat: float
    p_value: float
    frac_within_train95: float
    n_train: int
    n_gen: int


def mahalanobis(train_pieces: list[tuple[list[int], list[int]]],
                gen_pieces: list[tuple[list[int], list[int]]],
                reg: float = 1e-4) -> dict:
    """马氏距离报告 (DRMW 口径)。

    在训练集特征上拟合均值/协方差, 计算训练与生成样本的距离分布,
    用 Welch t 检验比较两组距离, 并报告生成样本落在训练距离 95 分位内的比例。
    """
    Xtr = np.array([melody_features(p, r) for p, r in train_pieces])
    Xge = np.array([melody_features(p, r) for p, r in gen_pieces])

    mu = Xtr.mean(axis=0)
    cov = np.cov(Xtr.T) + reg * np.eye(Xtr.shape[1])

    def dists(X):
        inv = np.linalg.inv(cov)
        d = X - mu
        return np.sqrt(np.einsum('ij,jk,ik->i', d, inv, d))

    dtr, dge = dists(Xtr), dists(Xge)
    t, p = stats.ttest_ind(dge, dtr, equal_var=False)

    thresh = np.percentile(dtr, 95)
    report = MahalanobisReport(
        train_mean=float(dtr.mean()), train_std=float(dtr.std()),
        gen_mean=float(dge.mean()), gen_std=float(dge.std()),
        t_stat=float(t), p_value=float(p),
        frac_within_train95=float((dge <= thresh).mean()),
        n_train=len(dtr), n_gen=len(dge),
    )
    return report.__dict__


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载辅助
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus_pieces(json_path: str, min_notes: int = 8):
    """从 melody JSON 加载乐曲 (按 prev_pc=null 切分) → [(pitches, rhythms)]。"""
    import json
    d = json.load(open(json_path, encoding='utf-8'))
    pieces, cur_p, cur_r = [], [], []
    for n in d:
        if n.get('prev_pc') is None and cur_p:
            pieces.append((cur_p, cur_r))
            cur_p, cur_r = [], []
        cur_p.append(n['pc'])
        cur_r.append(n['rhythm'])
    if cur_p:
        pieces.append((cur_p, cur_r))
    return [(p, r) for p, r in pieces if len(p) >= min_notes]


if __name__ == '__main__':
    pieces = load_corpus_pieces('data/processed/melody_llm_full_v2.json')
    # 训练集自身的马氏距离分布 (参考口径: 应接近训练均值)
    rep = mahalanobis(pieces, pieces)
    print('train-vs-train mahalanobis:', {k: round(v, 4) for k, v in rep.items()})
    # 训练集自身的 SIMAA (留一法近似: 与自己窗口的最大相似度)
    flat = [p for pcs, _ in pieces for p in pcs]
    gen = [pcs[:16] for pcs, _ in pieces if len(pcs) >= 16]
    s = simaa(flat, gen)
    print('train-vs-train SIMAA:', {k: round(v, 4) for k, v in s.items()})
