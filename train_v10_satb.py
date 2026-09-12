"""
v10 训练 —— 四声部切片扩散模型（众赞歌 + Palestrina）。

与 v9 的区别（一句话）：v9 学"一条旋律的下一个音"，v10 学"一个时间点上四个声部
怎么配合"。数据来自 `chorales_satb_v2.json`（368 首四声部众赞歌）与
`palestrina_satb_v1.json.gz`（1236 首文艺复兴复调，可选）。

表示
----
每首曲子 = 一串**纵向切片**；每个切片存四个声部的 (音高, 时值)：
    pitch  [V, T]  绝对 MIDI（36–88）或 REST
    rhythm [V, T]  时值 bin（0–7）或 HOLD（8，表示延音、不重新起音）
"是否在此起音"由 rhythm 是否为 HOLD 直接表达，不需要额外字段。

训练任务（每窗随机）
--------------------
    harmonize 40%  只给 1–2 个声部，补其余（经典配和声）
    infill    30%  挖掉一段连续时间的所有声部（续写）
    random    30%  逐 (声部, 切片) 随机挖

评估（关键：全部配人类语料基线）
--------------------
    - harmonize 任务的逐声部音高/节奏准确率（有真值）
    - 声部进行指标：平行五/八度、交越、间距、覆盖、音域违规
      —— 与**同一批窗口的真值切片**对比，而不是拍脑袋定阈值

用法
----
    # 本地快速验证（众赞歌，5 分钟）
    python train_v10_satb.py --corpus chorales --epochs 120 --win 48

    # 云端全量（众赞歌 + Palestrina，4090 上约 10 分钟）
    python train_v10_satb.py --corpus both --epochs 120 --d 480 --layers 8

    # 导出四声部 MIDI 试听
    python train_v10_satb.py --corpus chorales --export-only --ckpt v10.pt
"""
from __future__ import annotations

import argparse
import gzip
import json
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
from torch.utils.data import DataLoader, Dataset

from model.satb_diffusion import (N_VOICES, VOICE_NAMES, SATBDiffusion, SatbVocab,
                                  voiceleading_metrics)

DATA = ROOT / 'data/processed'
RHYTHM_VALUES = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]   # = MELODY_RHYTHM
HOLD = 8                       # 延音（不重新起音）
N_RHYTHM = 9                   # 0-7 时值 + 8 HOLD；MASK 用 id 9（见 SatbDiffusion）
CADENCE_ID = {'authentic': 1, 'half': 2, 'deceptive': 3, 'plagal': 4}
STYLE_ID = {'chorale': 0, 'palestrina': 1}
TPB = 480


def dur_to_bin(q: float) -> int:
    best, bd = 4, 1e9
    for i, v in enumerate(RHYTHM_VALUES):
        if abs(q - v) < bd:
            bd, best = abs(q - v), i
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def piece_to_window(piece: dict, vocab: SatbVocab, ws: int, we: int, trans: int = 0):
    """把一首曲子的 [ws, we) 切片区间转成模型输入张量（+转调）。"""
    slices = piece['slices'][ws:we]
    T = len(slices)
    if T < 4:
        return None
    pitch = np.full((N_VOICES, T), vocab.rest, dtype=np.int64)
    rhythm = np.full((N_VOICES, T), HOLD, dtype=np.int64)
    for t, s in enumerate(slices):
        for v, name in enumerate(VOICE_NAMES):
            m = s['midi'].get(name)
            if m is None:
                continue                                   # rest
            m2 = m + trans
            if not (vocab.lo <= m2 <= vocab.hi):
                return None                                # 转调越界 → 丢弃该窗
            pitch[v, t] = m2 - vocab.lo
            rhythm[v, t] = dur_to_bin(s['dur']) if s['attack'].get(name) else HOLD

    # 条件流（每切片）
    n_all = len(piece['slices'])
    base = ws
    style = STYLE_ID.get(piece.get('style', 'chorale'), 0)
    func = np.full(T, 7, dtype=np.int64)                   # 7 = 未知
    typ = np.full(T, 9, dtype=np.int64)                    # 9 = 未知
    root = np.full(T, 12, dtype=np.int64)
    if 'func' in slices[0]:                                # 众赞歌有功能和声标注
        for t, s in enumerate(slices):
            if s.get('func') is not None:
                func[t] = {'T': 0, 'PD': 1, 'D': 2, 'Sec': 3, 'Other': 4, 'REST': 5}.get(s['func'], 4)
                typ[t] = {'M': 0, 'm': 1, 'dim': 2, 'dom7': 3, 'aug': 4, 'm7': 5,
                          'dim7': 6, 'REST': 7}.get(s.get('type') or 'M', 0)
                root[t] = min(11, s.get('root') or 0)
    # 乐句流: 距本句末切片数（0=句末），封顶 5；6 = 未知
    phrase_bin = np.full(T, 6, dtype=np.int64)
    cadence = np.zeros(T, dtype=np.int64)
    ends = [ph['end_slice'] for ph in piece['phrases']]
    cads = [CADENCE_ID.get(ph.get('cadence') or 'authentic', 1) for ph in piece['phrases']]
    for t in range(T):
        gi = base + t
        nxt = next((k for k, e in enumerate(ends) if e >= gi), len(ends) - 1)
        phrase_bin[t] = min(max(0, ends[nxt] - gi), 5)
        cadence[t] = cads[nxt] if ends[nxt] >= gi else 0
    pos_bin = np.minimum((np.arange(T) + base) * 8 // max(n_all, 1), 7)
    rem = 1.0 - (np.arange(T) + base + 1) / max(n_all, 1)
    toend_bin = np.where(rem < 0.06, 0, np.where(rem < 0.15, 1, np.where(rem < 0.4, 2, 3)))
    offs = torch.tensor([float(x['off']) - float(slices[0]['off']) for x in slices],
                        dtype=torch.float)
    return {
        'offs': offs,
        'pitch': torch.tensor(pitch), 'rhythm': torch.tensor(rhythm),
        'func': torch.tensor(func), 'type': torch.tensor(typ), 'root': torch.tensor(root),
        'pos_bin': torch.tensor(pos_bin.astype(np.int64)),
        'toend_bin': torch.tensor(toend_bin.astype(np.int64)),
        'phrase_bin': torch.tensor(phrase_bin), 'cadence': torch.tensor(cadence),
        'style': torch.full((T,), style, dtype=torch.long),
        'piece': piece['id'], 'start': base,
    }


def build_windows(pieces, vocab, win: int, stride: int, limit: int = 0, transpose_aug: int = 0):
    out = []
    for p in pieces:
        T = len(p['slices'])
        if T < win:                     # 太短的作品跳过 (众赞歌 ~82 切片, Palestrina ~219)
            continue
        for ws in range(0, T - win + 1, stride):   # 只取**完整**窗口 → 长度统一
            we = ws + win
            for tr in ([0] if transpose_aug == 0 else
                       list(range(-transpose_aug, transpose_aug + 1))):
                w = piece_to_window(p, vocab, ws, we, tr)
                if w:
                    out.append(w)
    random.shuffle(out)
    return out[:limit] if limit else out


class WindowDataset(Dataset):
    KEYS = ['pitch', 'rhythm', 'func', 'type', 'root', 'pos_bin', 'toend_bin',
            'phrase_bin', 'cadence', 'style']

    def __init__(self, windows):
        self.w = windows

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return tuple(self.w[i][k] for k in self.KEYS)


def load_corpus(kind: str, vocab: SatbVocab, win: int, stride: int,
                transpose_aug: int = 0):
    pieces = []
    if kind in ('chorales', 'both'):
        pieces += json.load(open(DATA / 'chorales_satb_v2.json', encoding='utf-8'))
    if kind in ('palestrina', 'both'):
        with gzip.open(DATA / 'palestrina_satb_v1.json.gz', 'rt', encoding='utf-8') as fh:
            pieces += json.load(fh)
    print(f'语料 {kind}: {len(pieces)} 首, 切片 {sum(len(p["slices"]) for p in pieces):,}')
    return build_windows(pieces, vocab, win, stride, transpose_aug=transpose_aug)


def group_split(windows, val_ratio=0.08, seed=42):
    """按**曲目**分组划分（v8 的教训：滑窗不能按样本切）。"""
    ids = sorted({w['piece'] for w in windows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    tr = [w for w in windows if w['piece'] not in val_ids]
    vl = [w for w in windows if w['piece'] in val_ids]
    return tr, vl, val_ids


# ─────────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────────

def cond_of(batch, device, keys):
    return {k: batch[keys.index(k)].to(device) if k in keys else None for k in keys}


@torch.no_grad()
def eval_harmonize(model, windows, device, n=48, seed=0,
                   keep_voices=(0,)):
    """只给 keep_voices（默认 Soprano），生成其余声部 → 准确率 + 声部进行指标。

    分 **chorale（有功能和声条件）** 与 **palestrina（自由对位）** 两个口径报告：
    混在一起会稀释信号（后者本质更难）。声部进行指标配**同一批窗口的真值**作基线。
    """
    model.eval()
    rng = random.Random(seed)
    sample = windows if len(windows) <= n else rng.sample(windows, n)
    acc = {0: [0, 0], 1: [0, 0]}          # style → [pitch 命中, 实音数]
    acc_r = {0: [0, 0], 1: [0, 0]}
    gen_metrics, ref_metrics = Counter(), Counter()
    per_voice = {v: [0, 0] for v in range(N_VOICES)}
    for w in sample:
        style = int(w['style'][0].item())
        pitch = w['pitch'][None].to(device)
        rhythm = w['rhythm'][None].to(device)
        known = torch.zeros_like(pitch, dtype=torch.bool)
        for v in keep_voices:
            known[:, v] = True
        cond = {k: w[k][None].to(device) for k in
                ('func', 'type', 'root', 'pos_bin', 'toend_bin', 'phrase_bin', 'cadence', 'style')}
        # 单步 argmax = 与训练一致的用法 (实测 0.62 vs 迭代 16 步 0.14)
        gp, gr = model.generate(pitch, rhythm, known, cond, steps=1, argmax=True)
        tgt_p, tgt_r = pitch[0], rhythm[0]
        for v in range(N_VOICES):
            if v in keep_voices:
                continue
            real = (tgt_p[v] < model.vocab.n_pitch)
            hit = ((gp[0, v] == tgt_p[v]) & real).sum().item()
            acc[style][0] += hit; acc[style][1] += int(real.sum().item())
            acc_r[style][0] += (gr[0, v] == tgt_r[v]).sum().item()
            acc_r[style][1] += tgt_p[v].numel()
            per_voice[v][0] += hit; per_voice[v][1] += int(real.sum().item())
        # 声部进行指标（生成 vs 真值，同一批窗口）
        g_midi = torch.where(gp[0] < model.vocab.n_pitch, gp[0] + model.vocab.lo,
                             torch.full_like(gp[0], -1))
        t_midi = torch.where(tgt_p < model.vocab.n_pitch, tgt_p + model.vocab.lo,
                             torch.full_like(tgt_p, -1))
        for k, v in voiceleading_metrics(g_midi[None]).items():
            gen_metrics[k] += v
        for k, v in voiceleading_metrics(t_midi[None]).items():
            ref_metrics[k] += v
    pairs = max(gen_metrics['pairs'], 1)
    ref_pairs = max(ref_metrics['pairs'], 1)
    out = {
        'harm_pitch_acc_masked': (acc[0][0] + acc[1][0]) / max(acc[0][1] + acc[1][1], 1),
        'harm_rhythm_acc': (acc_r[0][0] + acc_r[1][0]) / max(acc_r[0][1] + acc_r[1][1], 1),
        'n_eval_windows': len(sample),
        'per_voice_pitch_acc': {VOICE_NAMES[v]: per_voice[v][0] / max(per_voice[v][1], 1)
                                for v in range(N_VOICES) if per_voice[v][1]},
        'parallel_5_per100': 100.0 * gen_metrics['parallel_5'] / pairs,
        'parallel_8_per100': 100.0 * gen_metrics['parallel_8'] / pairs,
        'crossing_per100': 100.0 * gen_metrics['crossing'] / pairs,
        'spacing_per100': 100.0 * gen_metrics['spacing'] / pairs,
        'range_viol_per100': 100.0 * gen_metrics['range_violation'] / pairs,
        'ref_parallel_5_per100': 100.0 * ref_metrics['parallel_5'] / ref_pairs,
        'ref_parallel_8_per100': 100.0 * ref_metrics['parallel_8'] / ref_pairs,
        'ref_crossing_per100': 100.0 * ref_metrics['crossing'] / ref_pairs,
        'ref_spacing_per100': 100.0 * ref_metrics['spacing'] / ref_pairs,
        'ref_range_viol_per100': 100.0 * ref_metrics['range_violation'] / ref_pairs,
    }
    for sid, name in ((0, 'chorale'), (1, 'palestrina')):
        if acc[sid][1]:
            out[f'harm_pitch_acc_{name}'] = acc[sid][0] / acc[sid][1]
            out[f'harm_rhythm_acc_{name}'] = acc_r[sid][0] / max(acc_r[sid][1], 1)
    return out


@torch.no_grad()
def eval_recovery(model, loader, device, mode='random'):
    """掩码恢复准确率（三种模式各测一次，衡量通用能力）。"""
    model.eval()
    out = {}
    for m in ('harmonize', 'infill', 'random'):
        cor = tot = cor_r = tot_r = 0
        for bi, b in enumerate(loader):
            if bi >= 6:
                break
            keys = WindowDataset.KEYS
            cond = {k: b[keys.index(k)].to(device) for k in keys}
            pitch = cond.pop('pitch'); rhythm = cond.pop('rhythm')
            torch.manual_seed(0)
            known = model.sample_mask(pitch.shape[0], pitch.shape[1], pitch.shape[2],
                                      m, pitch.device)
            pl, rl = model.forward(*[torch.where(
                known, x, torch.full_like(x, model.vocab.mask if x is pitch else model.num_rhythm))
                for x in (pitch, rhythm)], known, cond)
            tgt = ~known
            cor += (pl.argmax(-1)[tgt] == pitch[tgt]).sum().item()
            tot += int(tgt.sum().item())
            cor_r += (rl.argmax(-1)[tgt] == rhythm[tgt]).sum().item()
            tot_r += int(tgt.sum().item())
        out[f'rec_pitch_{m}'] = cor / max(tot, 1)
        out[f'rec_rhythm_{m}'] = cor_r / max(tot_r, 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MIDI 导出（试听用）
# ─────────────────────────────────────────────────────────────────────────────

def write_satb_midi(path, gen_pitch, gen_rhythm, offs, vocab, bpm=76,
                    programs=(52, 52, 52, 52)):
    """把 [V,T] 的生成结果写成四轨 MIDI（默认 program 52 = Choir Aahs）。

    offs: 每个切片在该窗口内的**起点(拍)**。起音 = rhythm != HOLD；
    时值取该音的时值 bin（延音切片不单独成音）。
    """
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    import mido
    mid = MidiFile(ticks_per_beat=TPB)
    meta = MidiTrack()
    meta.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    meta.append(MetaMessage('time_signature', numerator=4, denominator=4))
    mid.tracks.append(meta)
    V, T = gen_pitch.shape
    for v in range(V):
        tr = MidiTrack()
        tr.append(Message('program_change', program=programs[v], channel=v, time=0))
        notes = []                                    # (起始拍, midi, 时值)
        for t in range(T):
            p, r = gen_pitch[v, t].item(), gen_rhythm[v, t].item()
            if p >= vocab.n_pitch or r >= HOLD:       # REST / MASK / 延音
                continue
            notes.append([float(offs[t]), p + vocab.lo, RHYTHM_VALUES[min(r, 7)]])
        for i, n in enumerate(notes):                 # 时值不超过到下一个起音的距离
            if i + 1 < len(notes):
                n[2] = min(n[2], max(notes[i + 1][0] - n[0], 0.125))
        events = []
        for start, midi, dur in notes:
            on = int(round(start * TPB))
            events.append((on, 1, midi))
            events.append((on + max(int(round(dur * TPB)), 1), 0, midi))
        events.sort(key=lambda x: (x[0], x[1]))
        last = 0
        for tick, is_on, midi in events:
            tr.append(Message('note_on' if is_on else 'note_off', note=midi,
                              velocity=68 if is_on else 0, channel=v,
                              time=max(0, tick - last)))
            last = tick
        tr.append(MetaMessage('end_of_track'))
        mid.tracks.append(tr)
    mid.save(str(path))


@torch.no_grad()
def export_samples(model, windows, device, out_dir: Path, n=3, seed=0, bpm=76):
    """导出"给 Soprano 配四声部"的成品：生成版 + 真值版，便于 A/B 试听。"""
    model.eval()
    rng = random.Random(seed)
    pool = [w for w in windows if w['pitch'].shape[1] >= 16]
    picks = rng.sample(pool, min(n, len(pool)))
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for i, w in enumerate(picks):
        pitch = w['pitch'][None].to(device)
        rhythm = w['rhythm'][None].to(device)
        known = torch.zeros_like(pitch, dtype=torch.bool)
        known[:, 0] = True                                # 只给 Soprano
        cond = {k: w[k][None].to(device) for k in
                ('func', 'type', 'root', 'pos_bin', 'toend_bin', 'phrase_bin', 'cadence', 'style')}
        gp, gr = model.generate(pitch, rhythm, known, cond, steps=1, argmax=True)
        # 真值版（Soprano 用真值，其余声部用真值 — 即真实巴赫）
        offs = w['offs']
        write_satb_midi(out_dir / f'v10_gen_{i + 1}.mid', gp[0].cpu(), gr[0].cpu(),
                        offs, model.vocab, bpm)
        write_satb_midi(out_dir / f'v10_ref_{i + 1}.mid', pitch[0].cpu(), rhythm[0].cpu(),
                        offs, model.vocab, bpm)
        made += [f'v10_gen_{i + 1}.mid', f'v10_ref_{i + 1}.mid']
    return made


# ─────────────────────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['chorales', 'palestrina', 'both'], default='chorales')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--d', type=int, default=480)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--win', type=int, default=48)
    ap.add_argument('--stride', type=int, default=12)
    ap.add_argument('--aug-transpose', type=int, default=0,
                    help='随机移调半音数上限 (0=关闭; 2 表示 ±2)')
    ap.add_argument('--val-ratio', type=float, default=0.08)
    ap.add_argument('--out', type=str, default='v10_satb.pt')
    ap.add_argument('--tag', type=str, default='v10')
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--n-export', type=int, default=3)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = SatbVocab()
    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    if args.export_only:
        windows = load_corpus(args.corpus, vocab, args.win, args.stride)
        _, vl, _ = group_split(windows, args.val_ratio)
        model = SATBDiffusion(d=args.d, h=args.heads, layers=args.layers, vocab=vocab).to(device)
        model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device, weights_only=True))
        made = export_samples(model, vl, device, ROOT / 'data/generated', n=args.n_export)
        print('已导出:', made)
        return

    windows = load_corpus(args.corpus, vocab, args.win, args.stride,
                          transpose_aug=args.aug_transpose)
    tr_w, vl_w, val_ids = group_split(windows, args.val_ratio)
    print(f'窗口: 训练 {len(tr_w)} / 验证 {len(vl_w)} ({len(val_ids)} 首曲) | '
          f'序列长 {args.win} | 移调增广 ±{args.aug_transpose}')

    model = SATBDiffusion(d=args.d, h=args.heads, layers=args.layers, vocab=vocab).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'参数量 {n_par / 1e6:.1f}M | 设备 {device}')
    tr = DataLoader(WindowDataset(tr_w), batch_size=args.batch, shuffle=True, drop_last=True)
    vl = DataLoader(WindowDataset(vl_w), batch_size=args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    keys = WindowDataset.KEYS
    t0 = time.time()
    best = -1.0
    hist = []
    for ep in range(args.epochs):
        model.train()
        tot = nb = 0
        for b in tr:
            cond = {k: b[keys.index(k)].to(device) for k in keys}
            pitch = cond.pop('pitch'); rhythm = cond.pop('rhythm')
            opt.zero_grad()
            loss, lp, lr_, n_tgt = model.training_loss(pitch, rhythm, cond, mode='mixed')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            rec = eval_recovery(model, vl, device)
            harm = eval_harmonize(model, vl_w, device, n=24)
            score = (harm.get('harm_pitch_acc_chorale', harm['harm_pitch_acc_masked'])
                     + 0.3 * rec['rec_pitch_random'])
            line = (f'Ep{ep + 1:4d} | loss {tot / max(nb, 1):.4f} | '
                    f'rec(harm/infill/rand) {rec["rec_pitch_harmonize"]:.3f}/'
                    f'{rec["rec_pitch_infill"]:.3f}/{rec["rec_pitch_random"]:.3f} | '
                    f'配和声(众赞歌) 音高 {harm.get("harm_pitch_acc_chorale", 0):.3f} '
                    f'节奏 {harm.get("harm_rhythm_acc_chorale", 0):.3f} | '
                    f'平行五 {harm["parallel_5_per100"]:.1f} vs 真值 {harm["ref_parallel_5_per100"]:.1f} | '
                    f'{time.time() - t0:.0f}s')
            print(line, flush=True)
            hist.append({'epoch': ep + 1, 'loss': tot / max(nb, 1), **rec, **harm})
            if score > best:
                best = score
                torch.save(model.state_dict(), str(ROOT / args.out))
                print(f'   ↳ 保存检查点 (score {score:.3f})', flush=True)

    report = {'config': vars(args), 'n_params': n_par, 'n_train': len(tr_w), 'n_val': len(vl_w),
              'history': hist, 'best_score': best, 'seconds': round(time.time() - t0, 1)}
    json.dump(report, open(ROOT / f'data/processed/{args.tag}_report.json', 'w',
                           encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'→ data/processed/{args.tag}_report.json')
    if hist:
        h = hist[-1]
        print(f"最好成绩: 配和声音高 {h['harm_pitch_acc_masked']:.3f} | "
              f"逐声部 {h['per_voice_pitch_acc']}")

    made = export_samples(model, vl_w, device, ROOT / 'data/generated', n=args.n_export)
    print('试听样例:', made)


if __name__ == '__main__':
    main()
