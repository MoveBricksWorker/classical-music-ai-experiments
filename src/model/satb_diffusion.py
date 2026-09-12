"""
SATBDiffusion —— 四声部纵向切片上的非自回归离散扩散模型（v10）。

与 v9 (`melody_diffusion.MelodyDiffusion`) 的关系
------------------------------------------------
复用 v9 已验证的机制（D3PM 吸收态 [MASK]、MaskGIT 迭代解码、RoPE 相对位置），
把**建模单位从"一条旋律的音符"换成"一个纵向切片"**：

    v9 :  token = 一个旋律音           → 序列 = 单声部音符序列
    v10:  token = 一个时间点的四声部    → 序列 = 切片序列, 每个 token 含 4 个声部的
          (音高, 时值) 与"该声部已知/待生成"标记

模型第一次能看到"同时进行的几条线"，从而学声部进行、对位与织体 —
这是"听起来像真音乐"的根源，也是单声部表示学不到的东西。

三个刻意的设计决定
------------------------------------------------
1. **绝对音高而非音级**（默认 36–88 MIDI + 休止 + [MASK] = 55 类）：
   四声部写作里"低音在 C3 还是 C5"就是音乐本身；只用音级会丢掉八度信息，
   声部进行/间距/交越这些指标也算不出来。
2. **条件流相加而非拼接**：v9 用拼接（每流 d/流数 维），导致 d 必须被流数整除
   （加了声部/风格/织体流之后很难挑宽度）。这里各流独立嵌入到 d 维后**相加**。
3. **每声部独立输出头**：声部角色固定（S/A/T/B），比共享头更容易学到
   "低音要跳、女高音要唱"的差异。

训练时一个损失覆盖三种任务（每窗随机选一种掩码模式）
------------------------------------------------
  - `harmonize`（40%）：保留 1–2 个声部（通常含 Soprano），补其余声部
    —— 经典"众赞歌配和声"，最易评估、最直接可用；
  - `infill`  （30%）：挖掉一段连续切片上的所有声部 —— 续写/补全；
  - `random`  （30%）：逐 (声部, 切片) 随机挖 —— 通用掩码恢复。
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.melody_diffusion import RotaryEncoder

N_VOICES = 4
VOICE_NAMES = ['S', 'A', 'T', 'B']
PITCH_LO, PITCH_HI = 36, 88          # C2 – E6，覆盖 SATB 全部常用音域及 ±5 半音移调


class SatbVocab:
    """绝对音高词表: [PITCH_LO, PITCH_HI] + REST + MASK。"""

    def __init__(self, lo: int = PITCH_LO, hi: int = PITCH_HI):
        self.lo, self.hi = lo, hi
        self.n_pitch = hi - lo + 1
        self.rest = self.n_pitch
        self.mask = self.n_pitch + 1
        self.size = self.n_pitch + 2

    def encode(self, midi: torch.Tensor) -> torch.Tensor:
        """MIDI (int, 可能为 None/-1 表示休止) → 词表索引。"""
        idx = midi - self.lo
        idx = torch.where(midi < 0, torch.full_like(idx, self.rest), idx)
        return idx.clamp(0, self.size - 1)

    def decode(self, idx: torch.Tensor) -> torch.Tensor:
        """词表索引 → MIDI（rest/mask → -1）。"""
        return torch.where(idx < self.n_pitch, idx + self.lo,
                           torch.full_like(idx, -1))

    def is_pitch(self, idx: torch.Tensor) -> torch.Tensor:
        return idx < self.n_pitch


class SATBDiffusion(nn.Module):
    """四声部切片扩散模型。

    pitch/rhythm: [B, V, T]（V=4 声部, T=切片数）；每 (声部, 切片) 是一个建模单元。
    known:        [B, V, T] bool —— True = 给定条件（不参与损失、解码时保持）。
    """

    def __init__(self, d: int = 480, h: int = 8, layers: int = 8, max_len: int = 512,
                 n_voices: int = N_VOICES, vocab: SatbVocab | None = None,
                 num_funcs: int = 7, num_types: int = 9, num_rhythm: int = 9,
                 num_pos_bins: int = 8, num_toend_bins: int = 4, num_phrase_bins: int = 7,
                 num_cadence_types: int = 5, num_styles: int = 2,
                 dropout: float = 0.1, use_rope: bool = True):
        super().__init__()
        assert d % h == 0, f'd={d} 必须能被头数 h={h} 整除'
        self.d, self.V, self.max_len = d, n_voices, max_len
        self.vocab = vocab or SatbVocab()
        self.use_rope = use_rope
        self.num_rhythm = num_rhythm
        # 未知占位 id（比正常值大 1 位）
        self.unk = dict(func=num_funcs, type=num_types, root=12,
                        pos=num_pos_bins, toend=num_toend_bins,
                        phrase=num_phrase_bins, cadence=num_cadence_types)

        self.pitch_emb = nn.Embedding(self.vocab.size, d)
        self.rhythm_emb = nn.Embedding(num_rhythm + 1, d)
        self.known_emb = nn.Embedding(2, d)
        self.voice_emb = nn.Embedding(n_voices, d)

        self.func_emb = nn.Embedding(num_funcs + 1, d)
        self.type_emb = nn.Embedding(num_types + 1, d)
        self.root_emb = nn.Embedding(13, d)
        self.pos_emb = nn.Embedding(num_pos_bins + 1, d)
        self.toend_emb = nn.Embedding(num_toend_bins + 1, d)
        self.phrase_emb = nn.Embedding(num_phrase_bins + 1, d)
        self.cadence_emb = nn.Embedding(num_cadence_types + 1, d)
        self.style_emb = nn.Embedding(num_styles, d)

        self.drop = nn.Dropout(dropout)
        if use_rope:
            self.encoder = RotaryEncoder(d, h, layers, dropout=dropout, max_len=1024)
        else:
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=h, dim_feedforward=d * 4,
                                               dropout=dropout, batch_first=True,
                                               activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

        self.pitch_heads = nn.ModuleList([nn.Linear(d, self.vocab.size) for _ in range(n_voices)])
        self.rhythm_heads = nn.ModuleList([nn.Linear(d, num_rhythm + 1) for _ in range(n_voices)])

    # ─────────────────────────────────────────────────────────────
    def embed(self, pitch, rhythm, known, cond):
        B, V, T = pitch.shape
        T = min(T, self.max_len)
        pitch, rhythm, known = pitch[:, :, :T], rhythm[:, :, :T], known[:, :, :T]
        dev = pitch.device
        e = self.pitch_emb(pitch) + self.rhythm_emb(rhythm) + self.known_emb(known.long())
        e = e + self.voice_emb(torch.arange(V, device=dev))[None, :, None, :]
        e = e.sum(dim=1)                                    # 四声部相加 → [B, T, d]

        def add(emb, key, unk_id):
            v = cond.get(key)
            if v is None:
                v = torch.full((B, T), unk_id, dtype=torch.long, device=dev)
            return emb(v[:, :T])

        e = e + add(self.func_emb, 'func', self.unk['func'])
        e = e + add(self.type_emb, 'type', self.unk['type'])
        e = e + add(self.root_emb, 'root', self.unk['root'])
        e = e + add(self.pos_emb, 'pos_bin', self.unk['pos'])
        e = e + add(self.toend_emb, 'toend_bin', self.unk['toend'])
        e = e + add(self.phrase_emb, 'phrase_bin', self.unk['phrase'])
        e = e + add(self.cadence_emb, 'cadence', self.unk['cadence'])
        if 'style' in cond and cond['style'] is not None:
            e = e + self.style_emb(cond['style'][:, :T])
        return self.drop(e)

    def forward(self, pitch, rhythm, known, cond):
        T = min(pitch.shape[2], self.max_len)
        e = self.embed(pitch, rhythm, known, cond)
        o = self.encoder(e)                                          # [B, T, d]
        pl = torch.stack([h(o) for h in self.pitch_heads], dim=1)    # [B, V, T, P]
        rl = torch.stack([h(o) for h in self.rhythm_heads], dim=1)   # [B, V, T, R]
        return pl, rl

    # ─────────────────────────────────────────────────────────────
    def sample_mask(self, B: int, V: int, T: int, mode: str, dev) -> torch.Tensor:
        """known [B, V, T]（True = 已知）。mode: harmonize/infill/random/mixed。"""
        known = torch.zeros(B, V, T, dtype=torch.bool, device=dev)
        for b in range(B):
            m = mode
            if m == 'mixed':
                m = random.choices(['harmonize', 'infill', 'random'],
                                   weights=[0.4, 0.3, 0.3])[0]
            if m == 'harmonize':
                keep = {0} if random.random() < 0.7 else set()
                keep |= set(random.sample([1, 2, 3], k=random.choice([0, 1])))
                for v in keep:
                    known[b, v] = True
            elif m == 'infill':
                w = random.randint(max(1, T // 4), max(2, T // 2))
                s = random.randint(0, max(0, T - w))
                known[b, :, :] = True
                known[b, :, s:s + w] = False
            else:
                t = random.uniform(0.1, 0.9)
                known[b] = torch.rand(V, T, device=dev) >= t
                if bool(known[b].all()):
                    known[b, random.randrange(V), random.randrange(T)] = False
        return known

    def training_loss(self, pitch, rhythm, cond, mode='mixed', masks=None,
                      rhythm_weight: float = 0.3):
        known = masks if masks is not None else self.sample_mask(
            pitch.shape[0], pitch.shape[1], pitch.shape[2], mode, pitch.device)
        target = ~known
        p_in = torch.where(known, pitch, torch.full_like(pitch, self.vocab.mask))
        r_in = torch.where(known, rhythm, torch.full_like(rhythm, self.num_rhythm))
        pl, rl = self.forward(p_in, r_in, known, cond)
        lp = F.cross_entropy(pl[target], pitch[target])
        lr = F.cross_entropy(rl[target], rhythm[target])
        return lp + rhythm_weight * lr, lp.item(), lr.item(), int(target.sum().item())

    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, pitch, rhythm, known, cond, steps: int = 16, temp: float = 1.0,
                 remask_steps: int = 4, remask_ratio: float = 0.2, seed: int | None = None,
                 argmax: bool = False):
        """只生成 known=False 的单元；known=True 保持给定值（infilling 解码）。

        ⚠️ 实测（众赞歌配和声，只给 Soprano）：
            单步 + argmax  0.622  ← 与训练用法一致（一次预测全部掩码位）
            贪心 2/4/8 步  0.484 / 0.348 / 0.230
            采样 16 步     0.139
        **步数越多越差**：训练时 `known=True` 永远是**真值**，而迭代解码把自己的
        预测标成 known 喂回去 —— 训练/推理不一致（与 v5 条件块 bug 同源）。
        因此本模型的推荐解码是 `steps=1, argmax=True`；若要迭代精修，需先把
        "预测值"与"真值"用不同的 flag 区分（见 09 报告）。
        """
        self.eval()
        if seed is not None:
            torch.manual_seed(seed)
        B, V, T = pitch.shape
        T = min(T, self.max_len)
        pitch, rhythm = pitch[:, :, :T].clone(), rhythm[:, :, :T].clone()
        known = known[:, :, :T].clone()
        cond = {k: (v[:, :T] if torch.is_tensor(v) and v.dim() == 2 else v)
                for k, v in cond.items()}
        p_in = torch.where(known, pitch, torch.full_like(pitch, self.vocab.mask))
        r_in = torch.where(known, rhythm, torch.full_like(rhythm, self.num_rhythm))
        todo = int((~known).sum().item())
        if todo == 0:
            return p_in, r_in
        counts = [int(math.ceil(todo * math.cos(math.pi / 2 * (1 - (s + 1) / steps))))
                  for s in range(steps)]
        prev = 0
        for s in range(steps):
            cur_known = known | (p_in != self.vocab.mask)
            pl, rl = self.forward(p_in, r_in, cur_known, cond)
            pl_t = (pl / max(temp, 0.05)).clone()
            rl_t = (rl / max(temp, 0.05)).clone()
            pl_t[..., self.vocab.mask] = -float('inf')
            rl_t[..., self.num_rhythm] = -float('inf')
            p_conf = F.softmax(pl_t, dim=-1).max(-1).values
            r_conf = F.softmax(rl_t, dim=-1).max(-1).values
            conf = torch.where(~known, p_conf * r_conf, torch.full_like(p_conf, -1e9))

            if s < remask_steps and prev > 0:
                n_remask = int(prev * remask_ratio * (1 - s / max(remask_steps, 1)))
                done = (~known) & (p_in != self.vocab.mask)
                if n_remask > 0 and int(done.sum().item()) > 0:
                    d_conf = conf.masked_fill(~done, float('inf')).view(B, -1)
                    idx = d_conf.topk(min(n_remask, int(done.sum().item())), largest=False).indices
                    for b in range(B):
                        vv, tt = idx[b] // T, idx[b] % T
                        p_in[b, vv, tt] = self.vocab.mask
                        r_in[b, vv, tt] = self.num_rhythm
                    prev -= n_remask

            k = max(1, counts[s] - prev)
            flat = conf.view(B, -1)
            idx = flat.topk(min(k, int((flat > -1).sum().item())), dim=1).indices
            for b in range(B):
                vv, tt = idx[b] // T, idx[b] % T
                if argmax:
                    p_in[b, vv, tt] = pl_t[b, vv, tt].argmax(-1)
                    r_in[b, vv, tt] = rl_t[b, vv, tt].argmax(-1)
                else:
                    p_in[b, vv, tt] = torch.multinomial(F.softmax(pl_t[b, vv, tt], -1), 1).squeeze(-1)
                    r_in[b, vv, tt] = torch.multinomial(F.softmax(rl_t[b, vv, tt], -1), 1).squeeze(-1)
            prev = counts[s]
        return p_in, r_in


# ─────────────────────────────────────────────────────────────────────────────
# 音乐性指标（四声部写作），必须配人类语料基线才有意义
# ─────────────────────────────────────────────────────────────────────────────

def voiceleading_metrics(pitch_midi: torch.Tensor) -> dict:
    """pitch_midi: [B, V, T] 的 MIDI 值（-1 = 休止/未生成）。

    返回计数：parallel_5 / parallel_8（平行五/八度）、crossing（交越）、
    spacing（上三声部相邻间距 > 八度）、overlap（声部交叉覆盖）、
    range_violation（超常用音域）、pairs（参与统计的音程对数，用于归一化）。
    """
    B, V, T = pitch_midi.shape
    RANGES = [(60, 84), (55, 79), (48, 74), (40, 67)]     # S/A/T/B 常用音域
    out = dict(parallel_5=0, parallel_8=0, crossing=0, spacing=0, overlap=0,
               range_violation=0, pairs=0)
    pm = pitch_midi.to(torch.long)
    for b in range(B):
        for v in range(V):
            col = pm[b, v]
            for t in range(T):
                x = col[t].item()
                if x >= 0 and not (RANGES[v][0] <= x <= RANGES[v][1]):
                    out['range_violation'] += 1
        for v in range(V - 1):
            hi, lo = pm[b, v], pm[b, v + 1]
            prev_hi = prev_lo = None
            for t in range(T):
                a0, c0 = hi[t].item(), lo[t].item()
                if a0 < 0 or c0 < 0:
                    prev_hi = prev_lo = None
                    continue
                if c0 > a0:
                    out['crossing'] += 1
                if abs(a0 - c0) > 12 and v < 2:
                    out['spacing'] += 1
                if prev_hi is not None:
                    out['pairs'] += 1
                    # 声部交叉覆盖: 本声部越过上一时刻另一声部的位置
                    if lo[t].item() > prev_hi or hi[t].item() < prev_lo:
                        out['overlap'] += 1
                    da, dc = a0 - prev_hi, c0 - prev_lo
                    both_move = da != 0 and dc != 0
                    same_dir = (da > 0) == (dc > 0)
                    if both_move and same_dir:
                        ic_before = (prev_hi - prev_lo) % 12
                        ic_after = (a0 - c0) % 12
                        if ic_before == 7 and ic_after == 7:
                            out['parallel_5'] += 1          # 平行五度(含复合)
                        elif ic_before == 0 and ic_after == 0:
                            out['parallel_8'] += 1          # 平行八度/同度
                prev_hi, prev_lo = a0, c0
    return out
