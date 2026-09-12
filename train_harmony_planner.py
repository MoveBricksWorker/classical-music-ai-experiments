"""
和声规划器（planner）—— 从曲式模板生成**和弦计划**，再把计划交给实现器写四声部。

为什么需要它（v11"从零生成"失败的诊断）
------------------------------------------------
v11 让四声部模型在"全掩码 + 无和声标签"下从零写，结果只有 32% 的纵向音响
能构成合理和弦（人类 86%）——它必须在同一个注意力里同时决定"这里是什么和弦"
和"四个声部具体怎么落音"，而和弦标签此前一直是喂给它的（训练里 70% 的窗口都有）。

和声推断本身是**可监督的子任务**（语料里每个切片都有 func/type/root），
因此拆成两级更可能成功（也正是本项目"神经符号"的定位）：
    曲式模板 → **规划器（本文件）** → 和弦计划 → 实现器（v10/v12）→ 四个声部

模型
------------------------------------------------
小自回归 Transformer：每个切片一个 token，输入 = 上一切片的和弦嵌入 +
本切片的曲式流（位置/距尾/乐句距离/终止式类型/风格），因果掩码，
输出下一个切片的三元组 (func, type, root)。序列位置 0 是 BOS（用于预测首和弦）。

评估（有真值，可靠）
------------------------------------------------
- func / type / root 的逐切片准确率（按曲分组划分，多种子）
- 终止式落点：全终止计划处是否生成 T（主功能）和弦
- 和弦可解释率：预测的 (type, root) 反推出的和弦音，能否覆盖真实纵向音响

用法
------------------------------------------------
    python train_harmony_planner.py --epochs 200
    python train_harmony_planner.py --eval-only --ckpt harmony_planner.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import CHORD_INTERVALS
from train_v10_satb import DATA, piece_to_window
from model.satb_diffusion import SatbVocab

FUNC_NAMES = {0: 'T', 1: 'PD', 2: 'D', 3: 'Sec', 4: 'Other', 5: 'REST'}
ID2TYPE = {0: 'M', 1: 'm', 2: 'dim', 3: 'dom7', 4: 'aug', 5: 'm7', 6: 'dim7', 7: 'REST'}
N_FUNC, N_TYPE, N_ROOT = 8, 10, 13        # 含未知占位 (7 / 9 / 12)
N_FUNC_LABEL = len(FUNC_NAMES)            # 6 个真实功能标签 (0-5)
N_TYPE_LABEL = len(ID2TYPE)               # 8 个真实类型标签 (0-7)


def sample_plan_token(logits: torch.Tensor, n_labels: int, temp: float = 0.9) -> int:
    """从 logits 的**前 n_labels 类**采样一个和弦标签。

    未知占位（func=7 / type=9 / root=12）与不存在的 id（6/8）都要屏蔽：
    它们不是任何和弦，喂给实现器只会得到训练时从未出现过的条件
    （旧写法 `min(sample, N-2)` 会把它们夹成 6/8 这两个非法类别）。
    """
    lg = logits[:n_labels].clone() / max(temp, 1e-6)
    return int(torch.multinomial(torch.softmax(lg, -1), 1))


class HarmonyPlanner(nn.Module):
    def __init__(self, d: int = 256, h: int = 8, layers: int = 4, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        assert d % h == 0
        self.d, self.max_len = d, max_len
        self.func_emb = nn.Embedding(N_FUNC, d)
        self.type_emb = nn.Embedding(N_TYPE, d)
        self.root_emb = nn.Embedding(N_ROOT, d)
        self.bos = nn.Parameter(torch.zeros(d))
        self.sop_emb = nn.Embedding(64, d)          # Soprano 音高（词表索引 → 嵌入）
        self.pos_emb = nn.Embedding(9, d)
        self.toend_emb = nn.Embedding(5, d)
        self.phrase_emb = nn.Embedding(8, d)
        self.cadence_emb = nn.Embedding(6, d)
        self.style_emb = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=h, dim_feedforward=d * 4,
                                          dropout=dropout, batch_first=True,
                                          activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head_f = nn.Linear(d, N_FUNC)
        self.head_t = nn.Linear(d, N_TYPE)
        self.head_r = nn.Linear(d, N_ROOT)

    def plan_embed(self, cond):
        e = (self.pos_emb(cond['pos_bin']) + self.toend_emb(cond['toend_bin'])
             + self.phrase_emb(cond['phrase_bin']) + self.cadence_emb(cond['cadence'])
             + self.style_emb(cond['style']))
        return e

    def forward(self, cond, prev_func, prev_type, prev_root, sop=None):
        """cond: [B,T]；prev_*: [B,T] 的"上一切片/ BOS(-1)"；sop: [B,T] Soprano 音高索引。
        → 三个 logits [B,T,*]"""
        B, T = cond['pos_bin'].shape
        T = min(T, self.max_len)
        cond = {k: v[:, :T] for k, v in cond.items()}
        prev_func, prev_type, prev_root = prev_func[:, :T], prev_type[:, :T], prev_root[:, :T]
        e_prev = self.func_emb(prev_func.clamp(min=0)) + self.type_emb(prev_type.clamp(min=0)) \
            + self.root_emb(prev_root.clamp(min=0))
        bos_mask = (prev_func < 0).unsqueeze(-1)
        e_prev = torch.where(bos_mask, self.bos.expand_as(e_prev), e_prev)
        x = e_prev + self.plan_embed(cond)
        if sop is not None:
            x = x + self.sop_emb(sop[:, :T].clamp(0, 63))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        o = self.encoder(x, mask=mask)
        return self.head_f(o), self.head_t(o), self.head_r(o)


# ─────────────────────────────────────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────────────────────────────────────

def load_sequences(win: int = 128, stride: int = 16, limit: int = 0):
    """→ [(cond dict of [1,T] tensors, func[1,T], type[1,T], root[1,T], piece_id)]"""
    pieces = json.load(open(DATA / 'chorales_satb_v2.json', encoding='utf-8'))
    vocab = SatbVocab()
    out = []
    for p in pieces:
        T = len(p['slices'])
        if T < 16:
            continue
        for ws in range(0, max(1, T - win + 1), stride):
            we = min(ws + win, T)
            w = piece_to_window(p, vocab, ws, we, 0)
            if w is None:
                continue
            L = we - ws
            pad = win - L
            def _pd(x, v=0):
                return torch.cat([x, torch.full((pad,), v, dtype=x.dtype)]) if pad else x
            cond = {k: _pd(w[k])[None] for k in ('pos_bin', 'toend_bin', 'phrase_bin',
                                                 'cadence', 'style')}
            out.append((cond, _pd(w['func'])[None], _pd(w['type'])[None],
                        _pd(w['root'])[None], p['id'], L, _pd(w['pitch'][0])[None]))
    random.Random(0).shuffle(out)
    return out[:limit] if limit else out


def group_split(seqs, val_ratio=0.1, seed=42):
    ids = sorted({s[4] for s in seqs})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val = set(ids[:n_val])
    return [s for s in seqs if s[4] not in val], [s for s in seqs if s[4] in val], val


# ─────────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────────

def chord_explainable(type_id: int, root_id: int, true_pcs: set[int]) -> bool:
    if root_id >= 12 or type_id >= 8:
        return False
    tones = {(root_id + iv) % 12 for iv in CHORD_INTERVALS.get(ID2TYPE[type_id], [0, 4, 7])}
    # 预测和弦音能否覆盖真实纵向音响的 2/3 以上（允许经过音）
    if not true_pcs:
        return False
    return len(tones & true_pcs) / len(true_pcs) >= 0.66


@torch.no_grad()
def evaluate(model, seqs, device):
    model.eval()
    acc = {'func': [0, 0], 'type': [0, 0], 'root': [0, 0]}
    cad_hit = cad_tot = 0
    for cond, f, t, r, pid, seq_len, sop in seqs:
        cond = {k: v.to(device) for k, v in cond.items()}
        f, t, r = f.to(device), t.to(device), r.to(device)
        L = seq_len
        T = L
        pf = torch.cat([torch.full((1, 1), -1, device=device), f[:, :-1]], dim=1)
        pt = torch.cat([torch.full((1, 1), -1, device=device), t[:, :-1]], dim=1)
        pr = torch.cat([torch.full((1, 1), -1, device=device), r[:, :-1]], dim=1)
        lf, lt, lr = model(cond, pf, pt, pr, sop=sop.to(device))
        for name, logits, tgt in (('func', lf, f), ('type', lt, t), ('root', lr, r)):
            pred = logits.argmax(-1)[:, :L]          # 只算真实长度, 不把 padding 算进去
            acc[name][0] += (pred == tgt[:, :L]).sum().item()
            acc[name][1] += L
        # 终止式落点: 计划的全终止处 (cadence==1) 是否生成 T 功能
        cad = cond['cadence'][0]
        pf_pred = lf.argmax(-1)[0]
        for i in range(T):
            if cad[i].item() == 1:
                cad_tot += 1
                cad_hit += int(pf_pred[i].item() == 0)
    out = {f'{k}_acc': v[0] / max(v[1], 1) for k, v in acc.items()}
    out['cadence_T_rate'] = cad_hit / max(cad_tot, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--win', type=int, default=128)
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='harmony_planner.pt')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--ckpt', type=str, default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    seqs = load_sequences(args.win, args.stride)
    tr_seqs, vl_seqs, val_ids = group_split(seqs, 0.1, args.seed)
    print(f'片段: 训练 {len(tr_seqs)} / 验证 {len(vl_seqs)} ({len(val_ids)} 首曲) | '
          f'窗口长 {args.win}')

    model = HarmonyPlanner(d=args.d, layers=args.layers).to(device)
    print(f'规划器参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    if args.eval_only:
        model.load_state_dict(torch.load(ROOT / (args.ckpt or args.out),
                                        map_location=device, weights_only=True))
        r = evaluate(model, vl_seqs, device)
        print('验证: ' + ' | '.join(f'{k} {v:.3f}' for k, v in r.items()))
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = -1
    for ep in range(args.epochs):
        model.train()
        random.shuffle(tr_seqs)
        tot = nb = 0
        for i in range(0, len(tr_seqs), args.batch):
            batch = tr_seqs[i:i + args.batch]
            cond = {k: torch.cat([b[0][k] for b in batch]).to(device) for k in batch[0][0]}
            f = torch.cat([b[1] for b in batch]).to(device)
            t = torch.cat([b[2] for b in batch]).to(device)
            r = torch.cat([b[3] for b in batch]).to(device)
            lens = torch.tensor([b[5] for b in batch], device=device).unsqueeze(1)
            T = f.shape[1]
            m = (torch.arange(T, device=device).unsqueeze(0) < lens).reshape(-1)
            pf = torch.cat([torch.full((f.shape[0], 1), -1, device=device), f[:, :-1]], 1)
            pt = torch.cat([torch.full((f.shape[0], 1), -1, device=device), t[:, :-1]], 1)
            pr = torch.cat([torch.full((f.shape[0], 1), -1, device=device), r[:, :-1]], 1)
            lf, lt, lr = model(cond, pf, pt, pr, sop=torch.cat([b[6] for b in batch]).to(device))
            loss = (F.cross_entropy(lf.reshape(-1, N_FUNC), f.reshape(-1), reduction='none')[m].mean()
                    + F.cross_entropy(lt.reshape(-1, N_TYPE), t.reshape(-1), reduction='none')[m].mean()
                    + F.cross_entropy(lr.reshape(-1, N_ROOT), r.reshape(-1), reduction='none')[m].mean())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            r = evaluate(model, vl_seqs, device)
            score = r['func_acc'] + r['type_acc'] + r['root_acc']
            print(f'Ep{ep+1:4d} | loss {tot/max(nb,1):.4f} | func {r["func_acc"]:.3f} '
                  f'type {r["type_acc"]:.3f} root {r["root_acc"]:.3f} | '
                  f'全终止处生成T {r["cadence_T_rate"]:.3f}', flush=True)
            if score > best:
                best = score
                torch.save(model.state_dict(), str(ROOT / args.out))
    print(f'完成 → {args.out} (best {best:.3f})')


if __name__ == '__main__':
    main()
