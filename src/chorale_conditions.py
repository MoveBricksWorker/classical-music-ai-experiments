"""
众赞歌条件流编码 —— 训练 / 推理 / 评估的唯一实现。

为什么存在: 此前这套编码在 `train_v8_chorales.load_chorale_windows`、
`gen_sentence.piece_conditions` 里各写一份（`analyze_syntax` 又间接复用前者）。
项目第 1 条铁律是"训练/推理条件分布必须一致"，靠复制粘贴维护三份逻辑
迟早会重演 MelodyGPT 的条件块 bug —— 因此收敛到本模块。

口径（改动前先读，任何一处改动都会同时影响训练与推理）:
    - 和声条件 **4 音块多数投票恒定**: 语料 chordify 标注逐音变化，推理
      管线按块喂条件，训练必须同口径（v5 修复的内容）;
    - phrase_bin: 距本乐句末音数, 0=句末, 封顶 5, 6=未知占位;
    - cadence: 本乐句终止式类型 0=无/未知 1=全 2=半 3=阻碍 4=变格;
    - boundary: 1 = 该音为本乐句末（边界头监督信号）;
    - pos_bin: 全曲位置 8 桶; toend_bin: 距尾距离 4 桶。
"""
from __future__ import annotations

import json
import random
from collections import Counter

from constants import F2ID_MELODY, T2ID
from model.melody_diffusion import REST

WIN, STRIDE = 32, 6
BLOCK = 4

CADENCE_ID = {'authentic': 1, 'half': 2, 'deceptive': 3, 'plagal': 4}
CADENCE_NAME = {0: '弱分句', 1: '全终止', 2: '半终止', 3: '阻碍终止', 4: '变格终止'}

# 结构流未知占位: phrase_bin=6, cadence=0 (训练时随机丢弃结构流用)
PHRASE_UNKNOWN = 6


def block_constant(ids: list[int], block: int = BLOCK) -> list[int]:
    """每 block 个位置用多数投票压成常量（原地修改并返回）。"""
    n = len(ids)
    for b0 in range(0, n, block):
        stop = min(b0 + block, n)
        top = Counter(ids[b0:stop]).most_common(1)[0][0]
        ids[b0:stop] = [top] * (stop - b0)
    return ids


def piece_conditions(piece: dict, block: int = BLOCK,
                     unknown_struct: bool = False) -> dict:
    """一首众赞歌 → 逐音条件流。

    返回 dict of list: func/type/root/pc/rhythm/pos_bin/toend_bin/
    phrase_bin/cadence/boundary。unknown_struct=True 时把乐句/终止式
    结构流置为未知占位（评估边界头时用，避免泄漏答案）。
    """
    notes = piece['notes']
    n = len(notes)
    fids = block_constant([F2ID_MELODY.get(nt['func'], 4) for nt in notes], block)
    tids = block_constant([T2ID.get(nt['type'], 0) for nt in notes], block)
    rids = block_constant([min(11, nt['root']) for nt in notes], block)

    ends = {ph['note']: CADENCE_ID.get(ph['cadence'], 1) for ph in piece['phrases']}
    e_sorted = sorted(ends.keys())
    phrase_bin, cadence, boundary, pos_bin, toend_bin = [], [], [], [], []
    di, cur_end = 0, e_sorted[0]
    cur_cad = ends[cur_end]
    for i in range(n):
        while i > cur_end and di + 1 < len(e_sorted):
            di += 1
            cur_end = e_sorted[di]
            cur_cad = ends[cur_end]
        # 末句之后 (数据构造保证句末=末音, 此处为防御性 clamp) 视为句末
        phrase_bin.append(max(0, min(cur_end - i, 5)))
        cadence.append(0 if unknown_struct else cur_cad)
        boundary.append(1 if i == cur_end else 0)
        pos_bin.append(min(int(i / n * 8), 7))
        rem = 1.0 - (i + 1) / n
        toend_bin.append(0 if rem < 0.06 else (1 if rem < 0.15 else (2 if rem < 0.4 else 3)))
    if unknown_struct:
        phrase_bin = [PHRASE_UNKNOWN] * n
    return {
        'func': fids, 'type': tids, 'root': rids,
        'pc': [min(REST, nt['pc']) for nt in notes],
        'rhythm': [min(7, nt['rhythm']) for nt in notes],
        'pos_bin': pos_bin, 'toend_bin': toend_bin,
        'phrase_bin': phrase_bin, 'cadence': cadence, 'boundary': boundary,
    }


def phrase_spans(cond: dict, n: int | None = None) -> list[tuple[int, int]]:
    """由 phrase_bin 恢复乐句区间 [(start, end_inclusive), ...]。

    注意: unknown_struct 编码下 phrase_bin 全为 6, 不能用于恢复区间。
    """
    n = n if n is not None else len(cond['phrase_bin'])
    ends = [i for i in range(n) if cond['phrase_bin'][i] == 0]
    spans, s = [], 0
    for e in ends:
        spans.append((s, e))
        s = e + 1
    return spans


def load_chorale_windows(path: str, win: int = WIN, stride: int = STRIDE,
                         block: int = BLOCK) -> list[dict]:
    """数据集 → 滑窗样本。每个样本带 `piece` (曲索引) 便于按曲分组划分。"""
    pieces = json.load(open(path, encoding='utf-8'))
    samples = []
    for pi, p in enumerate(pieces):
        cond = piece_conditions(p, block)
        n = len(p['notes'])
        if n < win:
            continue
        for s in range(0, n - win + 1, stride):
            e = s + win
            samples.append({
                'piece': pi,
                'start': s,
                'func': cond['func'][s:e], 'type': cond['type'][s:e],
                'root': cond['root'][s:e], 'pc': cond['pc'][s:e],
                'rhythm': cond['rhythm'][s:e],
                'pos_bin': cond['pos_bin'][s:e], 'toend_bin': cond['toend_bin'][s:e],
                'phrase_bin': cond['phrase_bin'][s:e], 'cadence': cond['cadence'][s:e],
                'boundary': cond['boundary'][s:e],
            })
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 按曲分组划分 —— 修复"同曲窗口同时出现在训练/验证"的泄漏
# ─────────────────────────────────────────────────────────────────────────────

def grouped_split(samples: list[dict], val_ratio: float = 0.1,
                  seed: int = 42) -> tuple[list[dict], list[dict], set[int]]:
    """按曲分组划分 (GroupShuffleSplit 口径)。

    滑窗 win=32/stride=6 → 同曲相邻窗口重叠 81%，样本级随机划分会让
    96% 的验证窗口与训练窗口同曲重叠。本函数以 **曲** 为最小划分单位。
    返回 (train_samples, val_samples, val_piece_ids)。
    """
    piece_ids = sorted({s['piece'] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(piece_ids)
    n_val = max(1, int(round(len(piece_ids) * val_ratio)))
    val_pieces = set(piece_ids[:n_val])
    tr = [s for s in samples if s['piece'] not in val_pieces]
    vl = [s for s in samples if s['piece'] in val_pieces]
    return tr, vl, val_pieces


def kfold_by_piece(samples: list[dict], k: int = 5,
                   seed: int = 42) -> list[tuple[list[dict], list[dict], set[int]]]:
    """按曲 k 折交叉划分。返回 k 组 (train, val, val_pieces)。

    用途: 边界 F1 / 恢复准确率报告"多种子均值±标准差"，
    单次划分的波动（0.08-0.35）本身不可解释。
    """
    piece_ids = sorted({s['piece'] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(piece_ids)
    folds = [piece_ids[i::k] for i in range(k)]
    out = []
    for f in folds:
        val_pieces = set(f)
        tr = [s for s in samples if s['piece'] not in val_pieces]
        vl = [s for s in samples if s['piece'] in val_pieces]
        out.append((tr, vl, val_pieces))
    return out
