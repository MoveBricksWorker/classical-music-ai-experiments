"""
MelodyDiffusion —— GETMusic 式非自回归离散扩散旋律模型。

设计依据 (交接文档优先级①, 参考 GETMusic #5):
    - D3PM 吸收态 [MASK]: 训练时任意位置打掩码、模型恢复任意位置。
      "头"不再特殊 → Cadence 头失效的根因消失。
    - condition flags: 每个位置的 pitch/rhythm 有 真值/掩码 标志位,
      推理时模型显式知道"哪里已知、哪里待生成", 防止误差放大。
    - 推理用 MaskGIT 式置信度调度逐轮去掩码, 并支持 D3PM 风格
      回掩码 (remask) 允许早期错误被后续轮次纠正 —— 这是非自回归
      相比自回归的关键优势: 双向上下文 + 全局一致性。
    - 乐理 scorer 在解码轮次中按"模型置信度 × exp(scorer bonus)"
      重新加权候选, 保留项目"理论引导解码"的神经-符号定位
      (引导而非事后修补)。
    - 可选拍位条件 (num_beat_positions=4): 小节内拍位流 (0-3) 作为
      显式结构条件, 显著提升节奏恢复 (0.728 → 0.918)。

实验记录 (同数据 3,394 窗口 / 461 首):
    v1 无拍位 6.2M (d=252/8L): 恢复 0.736, SIMAA@90=0.000
    v2 无拍位 14M  (d=384/12L): 0.902, SIMAA@90=0.150 → 背谱, 弃用
    v3 拍位   14M  (d=378/12L): 0.924, SIMAA@90=0.217 → 背谱更重, 弃用
    v4 拍位   6.2M (d=252/8L): 0.746, SIMAA@90=0.017 → 拍位变体可选
    v5 块恒定和声条件 6.2M (d=252/8L): 训练条件按 4 音块多数投票恒定,
       与推理输入模式一致 —— 修复了"语料和声逐音变化 (连续长度=1 占 98%)
       vs 管线块重复"的训练/推理不一致 (完整管线振荡率 46%→23%)。
    v7 块恒定 + 结构流 + 乐句流 + RoPE 6.8M (d=288/8L): **当前管线默认**。
       phrase_bin (距本句末音数, 0=换气点) 来自 Ollama 本地标注
       (annotate_phrases.py, qwen3.5:9b, think=false)。乐句长度中位数 5。
       模型自发学到句末拉长 (4.35 vs 句中 3.85, 人类同模式 4.13/3.74);
       渲染层句末留 35% 气口。结尾质量: 末音=检测主音 88% (v6 38%),
       末音长音 88% (人类 64%)。振荡率 8.2% < 人类基线 15.9%。
    v6 块恒定 + 结构流 + RoPE 6.6M (d=256/8L):
       RoPE 相对位置替代可学习绝对位置 (训练窗口仅 16 音, 位置 16-191
       原为未训练随机嵌入); pos_bin(8)/toend_bin(4) 结构流从乐曲边界
       自动推导, 使模型感知"句中/句末"; 配合终止式后处理 (导音→主音,
       末小节短-短-短-长)。级进率 0.590→0.753, 马氏距离 6.64→5.02,
       末音落主音 100% / 长音 75% / 末 3 音含解决 100% (人类 64%/74%)。
    结论: 容量须与数据规模匹配; 条件分布对齐与位置泛化比容量更重要。

与 MelodyGPT (自回归) 的接口对齐:
    generate(func, typ, root, ...) → (gp[1,L], grh[1,L])

词表:
    pitch  : 0-11 音级, 12=REST, 13=[MASK]   (14 类)
    rhythm : 0-7 时值 (MELODY_RHYTHM), 8=[MASK] (9 类)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

PITCH_VOCAB = 14        # 0-11 pc, 12 REST, 13 MASK
RHYTHM_VOCAB = 9        # 0-7 rhythm, 8 MASK
PITCH_MASK = 13
RHYTHM_MASK = 8
REST = 12

# 结构流: 距结尾剩余比例 → 4 个 bin (0=结尾处, 3=远离结尾)
TOEND_BINS = 4


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE 旋转位置编码: x [B, h, L, dh], cos/sin [1, 1, L, dh/2]。"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class RotaryEncoderLayer(nn.Module):
    """自注意力 (RoPE 旋转位置, GETMusic 的 Roformer 口径) + FFN, 双向无掩码。"""

    def __init__(self, d: int, h: int, dropout: float = 0.1):
        super().__init__()
        self.h = h
        self.dh = d // h
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        hd = self.norm1(x)
        qkv = self.qkv(hd).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        o = (F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.out(o))
        x = x + self.ff(self.norm2(x))
        return x


class RotaryEncoder(nn.Module):
    """RoPE 编码器 —— 相对位置编码, 训练窗口长度之外可外推 (GETMusic 用 Roformer
    正是为了应对推理长度超出训练长度)。"""

    def __init__(self, d: int, h: int, num_layers: int,
                 dropout: float = 0.1, max_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.layers = nn.ModuleList([RotaryEncoderLayer(d, h, dropout)
                                     for _ in range(num_layers)])
        dh = d // h
        assert dh % 2 == 0, f'旋转位置要求 head_dim={dh} 为偶数'
        inv = 1.0 / (base ** (torch.arange(0, dh, 2).float() / dh))
        freqs = torch.outer(torch.arange(max_len).float(), inv)   # [L, dh/2]
        self.register_buffer('cos', freqs.cos()[None, None])
        self.register_buffer('sin', freqs.sin()[None, None])

    def forward(self, x):
        L = x.shape[1]
        cos, sin = self.cos[:, :, :L], self.sin[:, :, :L]
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


class MelodyDiffusion(nn.Module):
    """双向 Transformer 编码器 + pitch/rhythm 双头, 掩码恢复训练。

    可选拍位条件 (num_beat_positions=4): 额外输入小节内拍位流 (0-3),
    让模型显式学习"强拍放长音、弱拍放短音"的结构规律 —— 节奏对齐的
    正确解法 (评估发现调温度只是治标)。
    """

    def __init__(self, d: int = 256, h: int = 8, L: int = 8,
                 max_len: int = 256,
                 num_funcs: int = 7, num_types: int = 9,
                 pitch_classes: int = PITCH_VOCAB,
                 rhythm_classes: int = RHYTHM_VOCAB,
                 num_beat_positions: int | None = None,
                 num_pos_bins: int = 0,
                 num_toend_bins: int = 0,
                 num_phrase_bins: int = 0,
                 num_cadence_types: int = 0,
                 use_rope: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.num_funcs = num_funcs
        self.num_types = num_types
        self.num_beats = num_beat_positions
        self.num_pos_bins = num_pos_bins
        self.num_toend_bins = num_toend_bins
        self.num_phrase_bins = num_phrase_bins
        self.num_cadence_types = num_cadence_types
        self.use_rope = use_rope
        n_streams = (6 + (1 if num_beat_positions else 0)
                     + (1 if num_pos_bins else 0) + (1 if num_toend_bins else 0)
                     + (1 if num_phrase_bins else 0)
                     + (1 if num_cadence_types else 0))
        assert d % n_streams == 0 and d % h == 0, \
            f'd={d} 必须能被 {n_streams} (流数) 和 h={h} (头数) 整除'
        D = d // n_streams

        self.pitch_emb = nn.Embedding(pitch_classes, D)
        self.rhythm_emb = nn.Embedding(rhythm_classes, D)
        # 组合条件旗标 (GETMusic condition flags):
        #   0 = pitch/rhythm 均未知, 1 = pitch 已知, 2 = rhythm 已知, 3 = 均已知
        self.flag_emb = nn.Embedding(4, D)
        self.func_emb = nn.Embedding(num_funcs, D)
        self.type_emb = nn.Embedding(num_types, D)
        self.root_emb = nn.Embedding(12, D)
        if num_beat_positions:
            self.beat_emb = nn.Embedding(num_beat_positions, D)
        if num_pos_bins:
            # 结构流 1: 全曲位置 bin (起始 → 结尾)
            self.pos_emb_struct = nn.Embedding(num_pos_bins, D)
        if num_toend_bins:
            # 结构流 2: 距结尾剩余 bin (0=结尾处)
            self.toend_emb = nn.Embedding(num_toend_bins, D)
        if num_phrase_bins:
            # 乐句流: 距本乐句句末的音数 bin (0=句末换气点, 封顶 num_phrase_bins-1)
            self.phrase_emb = nn.Embedding(num_phrase_bins, D)
        if num_cadence_types:
            # 终止式流: 本乐句的终止式类型 (0=无/继续, 1=全终止, 2=半终止, 3=阻碍, 4=变格)
            self.cadence_emb = nn.Embedding(num_cadence_types, D)
        if not use_rope:
            self.pos_emb = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)

        if use_rope:
            self.encoder = RotaryEncoder(d, h, L, dropout=dropout, max_len=1024)
        else:
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=h,
                                               dim_feedforward=d * 4,
                                               dropout=dropout, batch_first=True,
                                               activation='gelu')
            self.encoder = nn.TransformerEncoder(layer, num_layers=L)

        self.pitch_head = nn.Linear(d, pitch_classes)
        self.rhythm_head = nn.Linear(d, rhythm_classes)
        # 边界预测头 (辅助任务): 每个位置是否为乐句末——"理解句子"的可操作定义
        self.boundary_head = nn.Linear(d, 2)

    # ─────────────────────────────────────────────────────────────
    # 前向
    # ─────────────────────────────────────────────────────────────

    def embed(self, pitch, rhythm, func, type_, root, flag, beat=None,
              pos_bin=None, toend_bin=None, phrase_bin=None, cadence=None):
        B, L = pitch.shape
        L = min(L, self.max_len)
        streams = [
            self.pitch_emb(pitch[:, :L]), self.rhythm_emb(rhythm[:, :L]),
            self.flag_emb(flag[:, :L]),
            self.func_emb(func[:, :L]), self.type_emb(type_[:, :L]),
            self.root_emb(root[:, :L]),
        ]
        if self.num_beats:
            if beat is None:
                # 默认: 窗口位置 mod 4 (推理时由调用方提供真实拍位)
                beat = (torch.arange(L, device=pitch.device).unsqueeze(0) % self.num_beats)
            streams.append(self.beat_emb(beat[:, :L]))
        if self.num_pos_bins:
            if pos_bin is None:
                pos_bin = torch.zeros(B, L, dtype=torch.long, device=pitch.device)
            streams.append(self.pos_emb_struct(pos_bin[:, :L]))
        if self.num_toend_bins:
            if toend_bin is None:
                toend_bin = torch.zeros(B, L, dtype=torch.long, device=pitch.device)
            streams.append(self.toend_emb(toend_bin[:, :L]))
        if self.num_phrase_bins:
            if phrase_bin is None:
                phrase_bin = torch.zeros(B, L, dtype=torch.long, device=pitch.device)
            streams.append(self.phrase_emb(phrase_bin[:, :L]))
        if self.num_cadence_types:
            if cadence is None:
                cadence = torch.zeros(B, L, dtype=torch.long, device=pitch.device)
            streams.append(self.cadence_emb(cadence[:, :L]))
        e = torch.cat(streams, dim=-1)
        if not self.use_rope:
            pos = torch.arange(L, device=pitch.device).unsqueeze(0).expand(B, -1)
            e = e + self.pos_emb(pos)
        return self.drop(e)

    def forward(self, pitch, rhythm, func, type_, root, flag, beat=None,
                pos_bin=None, toend_bin=None, phrase_bin=None, cadence=None,
                return_boundary=False):
        """→ (pitch_logits, rhythm_logits[, boundary_logits]) 各 [B, L, V]。"""
        L = min(pitch.shape[1], self.max_len)
        e = self.embed(pitch, rhythm, func, type_, root, flag, beat, pos_bin, toend_bin,
                       phrase_bin, cadence)
        o = self.encoder(e)
        if return_boundary:
            return self.pitch_head(o), self.rhythm_head(o), self.boundary_head(o)
        return self.pitch_head(o), self.rhythm_head(o)

    # ─────────────────────────────────────────────────────────────
    # D3PM 训练损失
    # ─────────────────────────────────────────────────────────────

    def training_loss(self, pitch, rhythm, func, type_, root,
                      mask_frac: float | None = None, beat=None,
                      pos_bin=None, toend_bin=None, phrase_bin=None,
                      cadence=None, boundary=None, boundary_weight: float = 0.3):
        """随机掩码 + 恢复损失 (仅在掩码位置计算 CE)。

        mask_frac=None 时每样本独立采样 t ~ U[0.05, 0.95] (GETMusic 口径)。
        """
        B, L = pitch.shape
        dev = pitch.device

        if mask_frac is None:
            t = torch.empty(B, device=dev).uniform_(0.05, 0.95)
        else:
            t = torch.full((B,), mask_frac, device=dev)
        mask = torch.rand(B, L, device=dev) < t.unsqueeze(1)

        p_in = pitch.clone()
        r_in = rhythm.clone()
        p_in[mask] = PITCH_MASK
        r_in[mask] = RHYTHM_MASK

        flag = (~mask).long() * 2 + (~mask).long()   # 未掩码=3(均已知), 掩码=0

        if boundary is not None:
            pl, rl, bl = self.forward(p_in, r_in, func, type_, root, flag, beat,
                                      pos_bin, toend_bin, phrase_bin, cadence,
                                      return_boundary=True)
            lb = F.cross_entropy(bl.reshape(-1, 2), boundary.reshape(-1))
        else:
            pl, rl = self.forward(p_in, r_in, func, type_, root, flag, beat,
                                  pos_bin, toend_bin, phrase_bin, cadence)
            lb = torch.tensor(0.0, device=dev)

        lp = F.cross_entropy(pl[mask], pitch[mask])
        lr = F.cross_entropy(rl[mask], rhythm[mask])
        return lp + 0.3 * lr + boundary_weight * lb, lp.item(), lr.item(), lb.item()

    # ─────────────────────────────────────────────────────────────
    # 非自回归迭代解码 (MaskGIT 调度 + D3PM 回掩码)
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, func, typ, root,
                 steps: int = 16, temp: float = 1.0,
                 remask_steps: int = 0, remask_ratio: float = 0.3,
                 scorer=None, scorer_steps: int = 4,
                 expand: bool = True, beat=None,
                 pos_bin=None, toend_bin=None, phrase_bin=None, cadence=None,
                 repeat_damp: float = 0.3,
                 seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """从全 [MASK] 出发迭代去掩码生成 (pitch, rhythm)。

        func/typ/root: [B, n_chords] 和声条件; expand=True 时按每和弦 4 音
        展开为 [B, 4*n_chords], expand=False 时视为已展开的 [B, L]。
        beat: 可选小节内拍位 [B, L] (0..num_beats-1), 缺省按位置 mod 拍位数。
        scorer(pc, chord_tones, prev_pc, step) → bonus, 与原管线同签名。
        repeat_damp: 精修轮中与前一音相同的候选概率乘数 (抑制持续嗡鸣;
        简单三和弦下模型对同音的原始概率过高, 0.3 不足以打破长同音串)。
        """
        self.eval()
        dev = func.device
        B = func.shape[0]

        if func.dim() == 1:
            func, typ, root = func.unsqueeze(0), typ.unsqueeze(0), root.unsqueeze(0)
        if expand:
            n_chords = func.shape[1]
            func = func.repeat_interleave(4, dim=1)[:4 * n_chords]
            typ = typ.repeat_interleave(4, dim=1)[:4 * n_chords]
            root = root.repeat_interleave(4, dim=1)[:4 * n_chords]
        L = func.shape[1]

        p = torch.full((B, L), PITCH_MASK, dtype=torch.long, device=dev)
        r = torch.full((B, L), RHYTHM_MASK, dtype=torch.long, device=dev)

        if seed is not None:
            torch.manual_seed(seed)

        # 每步去掩码数量: 余弦调度 (MaskGIT)
        def unmask_counts():
            out = []
            for s in range(steps):
                frac = math.cos(math.pi / 2 * (1 - (s + 1) / steps))
                out.append(int(math.ceil(L * frac)))
            return out  # 各步累计已解码目标数

        counts = unmask_counts()
        prev_decoded = 0

        for s in range(steps):
            flag = (p != PITCH_MASK).long() * 2 + (r != RHYTHM_MASK).long()
            pl, rl = self.forward(p, r, func, typ, root, flag, beat, pos_bin, toend_bin, phrase_bin, cadence)

            # 温度 + 置信度; MASK 类不允许被选为解码内容
            pl_t = pl / max(temp, 0.05)
            rl_t = rl / max(temp, 0.05)
            pl_t = pl_t.clone()
            rl_t = rl_t.clone()
            pl_t[..., PITCH_MASK] = -float('inf')
            rl_t[..., RHYTHM_MASK] = -float('inf')
            p_conf_raw = F.softmax(pl_t, dim=-1).max(-1).values   # [B,L] 真实置信度
            r_conf_raw = F.softmax(rl_t, dim=-1).max(-1).values

            # 只有 pitch 与 rhythm 均仍为 MASK 的位置参与去掩码选择
            both_masked = (p == PITCH_MASK) & (r == RHYTHM_MASK)
            joint = torch.where(both_masked,
                                p_conf_raw * r_conf_raw,
                                torch.full_like(p_conf_raw, -1e9))

            target = counts[s]
            n_unmask = max(1, target - prev_decoded)

            # 回掩码 (D3PM): 早期把已解码中的低置信位置重新打掩码
            if s < remask_steps and prev_decoded > 0:
                n_remask = int(prev_decoded * remask_ratio * (1 - s / max(remask_steps, 1)))
                if n_remask > 0:
                    d_conf = p_conf_raw.masked_fill(p == PITCH_MASK, float('inf'))
                    # 对已解码位置取最低置信
                    low = d_conf.topk(n_remask, largest=False).indices  # [B, k]
                    for b in range(B):
                        p[b, low[b]] = PITCH_MASK
                        r[b, low[b]] = RHYTHM_MASK
                    prev_decoded -= n_remask
                    n_unmask = max(1, target - prev_decoded)

            # 联合置信度 = pitch 置信 × rhythm 置信 (同位置联合决策)
            topk = joint.topk(min(n_unmask, (joint > -1).sum().item()), dim=1).indices

            for b in range(B):
                idxs = topk[b][:n_unmask]
                p_dist = F.softmax(pl_t[b, idxs], dim=-1)
                r_dist = F.softmax(rl_t[b, idxs], dim=-1)
                p[b, idxs] = torch.multinomial(p_dist, 1).squeeze(-1)
                r[b, idxs] = torch.multinomial(r_dist, 1).squeeze(-1)
            prev_decoded = target

        # ── 理论引导精修: 最后 scorer_steps 轮内, 对低置信位置重采样 ──
        if scorer is not None and scorer_steps > 0:
            # 和弦音表 [L]
            t_map = {0: [0, 4, 7], 1: [0, 3, 7], 2: [0, 3, 6], 3: [0, 4, 7, 10],
                     4: [0, 4, 8], 5: [0, 3, 7, 10], 6: [0, 3, 6, 9]}
            cts_all = [[(int(root[0, i]) + iv) % 12 for iv in
                        t_map.get(int(typ[0, i].item()), [0, 4, 7])]
                       for i in range(L)]

            for s in range(scorer_steps):
                flag = (p != PITCH_MASK).long() * 2 + (r != RHYTHM_MASK).long()
                pl, rl = self.forward(p, r, func, typ, root, flag, beat, pos_bin, toend_bin, phrase_bin, cadence)
                pl_t = pl / max(temp, 0.05)
                probs = F.softmax(pl_t, dim=-1)               # [B,L,V]

                # 对每个位置: 用已解码邻居做乐理评分, 重加权
                new_p = p.clone()
                for i in range(L):
                    pi = p[0, i].item()
                    if pi == PITCH_MASK or pi == REST:
                        continue
                    prev_pc = p[0, i - 1].item() if i > 0 and p[0, i - 1] != PITCH_MASK else pi
                    pr = probs[0, i].clone()
                    for cand in range(REST):  # 仅实音候选
                        bonus = scorer(int(cand), cts_all[i], prev_pc, i)
                        pr[cand] = pr[cand] * math.exp(bonus * 0.5)
                    # 同音衰减: 抑制与前一音完全相同的候选 (理论引导而非事后修补)
                    if 0 <= prev_pc < REST:
                        pr[prev_pc] = pr[prev_pc] * repeat_damp
                    new_p[0, i] = torch.multinomial(pr[:REST] / pr[:REST].sum(), 1)
                p = new_p

        return p, r

    @torch.no_grad()
    def generate_d3pm(self, func, typ, root,
                      steps: int = 100, temp: float = 1.0,
                      scorer=None, scorer_steps: int = 0,
                      expand: bool = True, beat=None,
                      pos_bin=None, toend_bin=None, phrase_bin=None, cadence=None,
                      seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """GETMusic 口径的 D3PM 祖先采样 (吸收态 [MASK], x0 参数化)。

        与 MaskGIT 硬调度的区别: 每步对仍为 [MASK] 的位置按转移矩阵
        后验概率 p_keep(t) = γ_t·ᾱ_{t-1}/γ̄_t 随机解掩 (而非按置信度
        取固定数量), 更接近原论文的采样过程。
        调度: ᾱ_t = cos²(π t / 2T), ᾱ_0=1, ᾱ_T=0。T=100 为论文设置。
        """
        self.eval()
        dev = func.device
        if func.dim() == 1:
            func, typ, root = func.unsqueeze(0), typ.unsqueeze(0), root.unsqueeze(0)
        if expand:
            n_chords = func.shape[1]
            func = func.repeat_interleave(4, dim=1)[:4 * n_chords]
            typ = typ.repeat_interleave(4, dim=1)[:4 * n_chords]
            root = root.repeat_interleave(4, dim=1)[:4 * n_chords]
        L = func.shape[1]
        B = func.shape[0]

        if seed is not None:
            torch.manual_seed(seed)

        T = steps
        abar = [math.cos(math.pi * t / (2 * T)) ** 2 for t in range(T + 1)]
        abar[T] = 0.0

        p = torch.full((B, L), PITCH_MASK, dtype=torch.long, device=dev)
        r = torch.full((B, L), RHYTHM_MASK, dtype=torch.long, device=dev)

        # 和弦音表 (scorer 用)
        t_map = {0: [0, 4, 7], 1: [0, 3, 7], 2: [0, 3, 6], 3: [0, 4, 7, 10],
                 4: [0, 4, 8], 5: [0, 3, 7, 10], 6: [0, 3, 6, 9]}
        cts_all = [[(int(root[0, i]) + iv) % 12 for iv in
                    t_map.get(int(typ[0, i].item()), [0, 4, 7])] for i in range(L)]

        for t in range(T, 0, -1):
            flag = (p != PITCH_MASK).long() * 2 + (r != RHYTHM_MASK).long()
            pl, rl = self.forward(p, r, func, typ, root, flag, beat, pos_bin, toend_bin, phrase_bin, cadence)
            pl_t = pl / max(temp, 0.05)
            rl_t = rl / max(temp, 0.05)
            pl_t = pl_t.clone(); rl_t = rl_t.clone()
            pl_t[..., PITCH_MASK] = -float('inf')
            rl_t[..., RHYTHM_MASK] = -float('inf')

            # 后验解掩概率
            alpha_t = abar[t] / max(abar[t - 1], 1e-8)
            gamma_t = 1.0 - alpha_t
            gbar_t = 1.0 - abar[t]
            p_keep = (gamma_t * abar[t - 1]) / max(gbar_t, 1e-8)
            p_keep = min(max(p_keep, 0.0), 1.0)

            masked = (p == PITCH_MASK) & (r == RHYTHM_MASK)
            if not masked.any():
                break
            # 采样候选
            for b in range(B):
                idxs = masked[b].nonzero(as_tuple=True)[0]
                if len(idxs) == 0:
                    continue
                p_dist = F.softmax(pl_t[b, idxs], dim=-1)
                r_dist = F.softmax(rl_t[b, idxs], dim=-1)
                cand_p = torch.multinomial(p_dist, 1).squeeze(-1)
                cand_r = torch.multinomial(r_dist, 1).squeeze(-1)
                # 乐理 scorer 引导候选选择 (若提供)
                if scorer is not None and scorer_steps > 0:
                    for k, i in enumerate(idxs.tolist()):
                        pr = p_dist[k].clone()
                        prev_pc = p[0, i - 1].item() if i > 0 and p[0, i - 1] != PITCH_MASK else int(cand_p[k])
                        for c in range(REST):
                            bonus = scorer(int(c), cts_all[i], prev_pc, i)
                            pr[c] = pr[c] * math.exp(bonus * 0.5)
                        if 0 <= prev_pc < REST:
                            pr[prev_pc] = pr[prev_pc] * 0.3
                        cand_p[k] = torch.multinomial(pr[:REST] / pr[:REST].sum(), 1)
                keep = torch.rand(len(idxs), device=dev) < p_keep
                sel = idxs[keep]
                p[b, sel] = cand_p[keep]
                r[b, sel] = cand_r[keep]

            # 最后一步 (t=1, p_keep=1) 保证全部解掩
            if t == 1:
                still = (p == PITCH_MASK) | (r == RHYTHM_MASK)
                if still.any():
                    flag = (p != PITCH_MASK).long() * 2 + (r != RHYTHM_MASK).long()
                    pl, rl = self.forward(p, r, func, typ, root, flag, beat, pos_bin, toend_bin, phrase_bin, cadence)
                    pl[..., PITCH_MASK] = -float('inf')
                    rl[..., RHYTHM_MASK] = -float('inf')
                    p[still] = pl.argmax(-1)[still]
                    r[still] = rl.argmax(-1)[still]

        return p, r

    @torch.no_grad()
    def generate_with_context(self, func, typ, root,
                              init_pitch, init_rhythm, keep,
                              steps: int = 16, temp: float = 1.0,
                              scorer=None, scorer_steps: int = 0,
                              beat=None, pos_bin=None, toend_bin=None,
                              phrase_bin=None, cadence=None, repeat_damp: float = 0.05,
                              seed: int | None = None):
        """上下文补全生成 (动机复用): keep 位置保持 init 真值, 其余从 [MASK] 恢复。

        用于 A-B-A' 曲式: 目标乐句用前面乐句的素材 (转到当前和声的和弦音上)
        作为真值上下文, 模型补齐其余音 —— 既保留动机又适配新和声。
        func/typ/root: [B, L] 已是全曲展开的条件; init/keep 同为 [B, L]。
        """
        self.eval()
        dev = func.device
        B, L = func.shape
        p = torch.where(keep, init_pitch, torch.full_like(init_pitch, PITCH_MASK))
        r = torch.where(keep, init_rhythm, torch.full_like(init_rhythm, RHYTHM_MASK))
        if seed is not None:
            torch.manual_seed(seed)

        def unmask_counts():
            out = []
            for s_ in range(steps):
                frac = math.cos(math.pi / 2 * (1 - (s_ + 1) / steps))
                out.append(int(math.ceil(L * frac)))
            return out

        counts = unmask_counts()
        prev_decoded = int(keep.sum().item())
        for s_ in range(steps):
            flag = (p != PITCH_MASK).long() * 2 + (r != RHYTHM_MASK).long()
            pl, rl = self.forward(p, r, func, typ, root, flag, beat,
                                  pos_bin, toend_bin, phrase_bin)
            pl_t = pl / max(temp, 0.05)
            rl_t = rl / max(temp, 0.05)
            pl_t = pl_t.clone(); rl_t = rl_t.clone()
            pl_t[..., PITCH_MASK] = -float('inf')
            rl_t[..., RHYTHM_MASK] = -float('inf')
            p_conf = F.softmax(pl_t, dim=-1).max(-1).values
            r_conf = F.softmax(rl_t, dim=-1).max(-1).values
            both_masked = (p == PITCH_MASK) & (r == RHYTHM_MASK) & (~keep)
            joint = torch.where(both_masked, p_conf * r_conf,
                                torch.full_like(p_conf, -1e9))
            target = max(counts[s_], prev_decoded)
            n_unmask = max(1, target - prev_decoded)
            topk = joint.topk(min(n_unmask, (joint > -1).sum().item()), dim=1).indices
            for b in range(B):
                idxs = topk[b][:n_unmask]
                if len(idxs) == 0:
                    continue
                p[b, idxs] = torch.multinomial(F.softmax(pl_t[b, idxs], dim=-1), 1).squeeze(-1)
                r[b, idxs] = torch.multinomial(F.softmax(rl_t[b, idxs], dim=-1), 1).squeeze(-1)
            prev_decoded = target
        return p, r

    @torch.no_grad()
    def resample(self, pitch, rhythm, func, typ, root,
                 start: int, end: int,
                 steps: int = 8, temp: float = 1.0,
                 beat=None, pos_bin=None, toend_bin=None, phrase_bin=None, cadence=None,
                 seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """DeepBach 式局部重采样 (Briot & Pachet #18)。

        pitch/rhythm: [1, L] 已解码序列 (全真值);
        [start, end) 区间被重新打掩码, 其余位置保持真值作为双向上下文,
        迭代去掩码只更新区间内位置 → "改一个和弦, 只重生成对应小节"。
        """
        self.eval()
        dev = pitch.device
        B, L = pitch.shape
        p = pitch.clone()
        r = rhythm.clone()
        span = end - start
        p[:, start:end] = PITCH_MASK
        r[:, start:end] = RHYTHM_MASK

        if seed is not None:
            torch.manual_seed(seed)

        # 区间内位置按余弦调度去掩码
        counts = [int(math.ceil(span * math.cos(math.pi / 2 * (1 - (s + 1) / steps))))
                  for s in range(steps)]
        prev = 0
        for s in range(steps):
            flag = (p != PITCH_MASK).long() * 2 + (r != RHYTHM_MASK).long()
            pl, rl = self.forward(p, r, func, typ, root, flag, beat, pos_bin, toend_bin, phrase_bin, cadence)
            pl_t = pl / max(temp, 0.05)
            rl_t = rl / max(temp, 0.05)
            pl_t = pl_t.clone()
            rl_t = rl_t.clone()
            pl_t[..., PITCH_MASK] = -float('inf')
            rl_t[..., RHYTHM_MASK] = -float('inf')
            p_conf = F.softmax(pl_t, dim=-1).max(-1).values
            r_conf = F.softmax(rl_t, dim=-1).max(-1).values
            joint = p_conf * r_conf

            # 只允许区间内位置被解码
            region = torch.zeros_like(joint)
            region[:, start:end] = joint[:, start:end]
            region = region.masked_fill(~((p == PITCH_MASK) & (r == RHYTHM_MASK)), -1e9)

            n_unmask = max(1, counts[s] - prev)
            topk = region.topk(min(n_unmask, (region > -1).sum().item()), dim=1).indices
            for b in range(B):
                idxs = topk[b][:n_unmask]
                p[b, idxs] = torch.multinomial(F.softmax(pl_t[b, idxs], dim=-1), 1).squeeze(-1)
                r[b, idxs] = torch.multinomial(F.softmax(rl_t[b, idxs], dim=-1), 1).squeeze(-1)
            prev = counts[s]

        return p, r


if __name__ == '__main__':
    # 冒烟测试: 随机初始化的模型做一次前后向 + 一次生成
    torch.manual_seed(0)
    m = MelodyDiffusion(d=60, h=4, L=2, max_len=64)
    B, L = 2, 16
    pitch = torch.randint(0, 13, (B, L))
    rhythm = torch.randint(0, 8, (B, L))
    func = torch.randint(0, 7, (B, L))
    typ = torch.randint(0, 9, (B, L))
    root = torch.randint(0, 12, (B, L))
    loss, lp, lr = m.training_loss(pitch, rhythm, func, typ, root)
    print(f'loss={loss.item():.3f} (pitch={lp:.3f}, rhythm={lr:.3f})')

    fc = torch.randint(0, 7, (1, 8))
    tc = torch.randint(0, 9, (1, 8))
    rc = torch.randint(0, 12, (1, 8))
    gp, grh = m.generate(fc, tc, rc, steps=8, temp=1.2, seed=1)
    print('gen pitch:', gp.tolist())
    print('gen rhythm:', grh.tolist())
    print('params:', sum(p.numel() for p in m.parameters()))
