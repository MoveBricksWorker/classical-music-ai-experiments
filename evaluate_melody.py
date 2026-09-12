"""
旋律模型评估 —— 不看"掩码恢复率"，看**生成的旋律像不像人写的**。

自由生成的旋律没有唯一正确答案，所以逐音准确率无意义。这里用音乐性指标，
全部配**真众赞歌 Soprano 的语料基线**（同一套口径）：

    级进率 step_ratio      —— 相邻音程 ≤2 半音的比例（众赞歌高，跳进少）
    平均音程 / 方向变化率   —— 旋律轮廓的平稳度
    音域跨度               —— 是否落在女高音的常用音区
    乐句末落音             —— 是否落在稳定音级（1/5，即 C/G）
    末音是否主音           —— 整曲收束感
    起音密度               —— 每拍的起音数（太密=碎，太疏=空）

用法:
    python evaluate_melody.py --ckpt v14_melody.pt --melody-d 384 --melody-layers 6 --n 8
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from gen_satb import write_midi
from gen_from_scratch import build_plan, VOCAB
from model.satb_diffusion import SATBDiffusion
from train_v10_satb import (COND_KEYS, DATA, HOLD, RHYTHM_VALUES, group_split,
                            load_corpus, piece_to_window)

MD = ROOT        # 脚本所在目录（本地与云端通用）


def corpus_baseline() -> dict:
    """真众赞歌 Soprano 的语料基线。"""
    pieces = json.load(open(DATA / 'chorales_satb_v2.json', encoding='utf-8'))
    steps, ints, ranges, dens, ends, last_tonic, phr_end_stable = [], [], [], [], [], 0, []
    dirchg = []
    for p in pieces:
        pcs, durs, onsets = [], [], []
        for s in p['slices']:
            m = s['midi'].get('S')
            if m is None:
                continue
            pcs.append(m)
            onsets.append(s['off'])
            if s['attack'].get('S'):
                durs.append(s['dur'])
        if len(pcs) < 8:
            continue
        iv = [abs(b - a) for a, b in zip(pcs[:-1], pcs[1:])]
        steps += [x for x in iv if x <= 2]
        ints += iv
        ranges.append(max(pcs) - min(pcs))
        dens.append(len(durs) / max(sum(durs), 1e-6))
        if pcs[-1] % 12 == 0:
            last_tonic += 1
        for ph in p['phrases']:
            j = ph['end_slice']
            m = p['slices'][j]['midi'].get('S') if j < len(p['slices']) else None
            if m is not None:
                phr_end_stable.append(m % 12 in (0, 7))
        signs = [np.sign(b - a) for a, b in zip(pcs[:-1], pcs[1:]) if b != a]
        if len(signs) > 1:
            dirchg.append(sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b) / (len(signs) - 1))
    return {
        'step_ratio': len(steps) / max(len(ints), 1),
        'avg_interval': statistics.mean(ints) if ints else 0,
        'direction_change': statistics.mean(dirchg) if dirchg else 0,
        'range_span': statistics.mean(ranges) if ranges else 0,
        'onset_density': statistics.mean(dens) if dens else 0,
        'last_note_tonic': last_tonic / max(len(pieces), 1),
        'phrase_end_stable': sum(phr_end_stable) / max(len(phr_end_stable), 1),
        'n_pieces': len(pieces),
    }


MELODY_SCORER_DOC = """
旋律乐理评分（解码侧引导，与 v9 同思路）：
  +0.40 级进（1-2 半音）      -0.40 大跳（>=6 半音）
  +0.20 和弦音(C-E-G)         -0.30 与前一音相同（防振荡）
  乐句末（phrase_bin==0）额外：+0.60 落在稳定音级(1/3/5)，末音 +1.20 落主音 C
"""


def score_melody(cand_pc: int, prev_pc: int | None, is_phrase_end: bool,
                 is_last: bool) -> float:
    s = 0.0
    if cand_pc in (0, 4, 7):
        s += 0.20
    if prev_pc is not None:
        d = min(abs(cand_pc - prev_pc), 12 - abs(cand_pc - prev_pc))
        if 1 <= d <= 2:
            s += 0.40
        elif d >= 6:
            s -= 0.40
        if d == 0:
            s -= 0.30
    if is_phrase_end:
        s += 0.60 if cand_pc in (0, 4, 7) else -0.30
    if is_last:
        s += 1.20 if cand_pc == 0 else -0.60
    return s


@torch.no_grad()
def gen_melody_scored(model, cond, T, device, seed=0, temp=0.8, scorer_weight=1.0):
    """乐理评分引导的旋律解码（**在绝对音高上评分**，否则会到处跨八度）。

    评分（与 v9 的 scorer 同思路）：
      +0.40 级进(1-2 半音) / -0.40 大跳(>=6) / -0.30 同音重复（防振荡）
      +0.20 和弦音(C-E-G)；乐句末 +0.60 落稳定音级(1/3/5)；末音 +1.20 落主音

    ⚠️ 待生成位置喂 **[MASK]/num_rhythm**（与训练、`generate()` 一致）。
    2026-09-12 修正：此前这里填的是 REST/HOLD，与训练约定不符
    （实测两条输入路径的 argmax 仅 28% 一致），当时的评分引导数字是分布外算的。
    """
    import torch.nn.functional as F
    torch.manual_seed(seed)
    pitch = torch.full((1, 1, T), VOCAB.mask, dtype=torch.long, device=device)
    rhythm = torch.full((1, 1, T), model.num_rhythm, dtype=torch.long, device=device)
    known = torch.zeros((1, 1, T), dtype=torch.bool, device=device)
    pl, rl = model.forward(pitch, rhythm, known, {k: cond[k] for k in COND_KEYS})
    probs = F.softmax(pl[0, 0] / temp, dim=-1).cpu().numpy()          # [T, 55]
    probs[:, VOCAB.rest:] = 0
    pe = cond['phrase_bin'][0].cpu().numpy()
    mids = np.arange(VOCAB.n_pitch) + VOCAB.lo                        # 绝对音高
    pcs = mids % 12
    stable = np.isin(pcs, (0, 4, 7))
    out = np.full(T, VOCAB.rest, dtype=np.int64)
    prev = None
    for t in range(T):
        sc = np.where(stable, 0.20, 0.0)
        if prev is not None:
            d = np.abs(mids - prev)
            sc = sc + np.where((d >= 1) & (d <= 2), 0.40, 0.0)                     - np.where(d >= 6, 0.40, 0.0) - np.where(d == 0, 0.30, 0.0)
        if pe[t] == 0:
            sc = sc + np.where(stable, 0.60, -0.30)
        if t == T - 1:
            sc = sc + np.where(pcs == 0, 1.20, -0.60)
        w = probs[t, :VOCAB.n_pitch] * np.exp(scorer_weight * sc)
        if not np.isfinite(w).all() or w.sum() <= 1e-12:
            w = probs[t, :VOCAB.n_pitch] + 1e-9
        out[t] = np.random.default_rng(seed * 1000 + t).choice(
            VOCAB.n_pitch, p=w / w.sum())
        prev = mids[out[t]]
    r = rl[0, 0].argmax(-1)                                           # 节奏仍用模型 argmax
    return torch.tensor(out, device=device)[None], r


@torch.no_grad()
def gen_melodies_with_harmony(model, n, seed, device, temp=0.9, win=64):
    """诊断模式: 用**真值曲子的和声+乐句条件**生成旋律（和声锚定），
    测量"有了和声的旋律"是否更像众赞歌。"""
    import json as _json
    pieces = _json.load(open(DATA / 'chorales_satb_v2.json', encoding='utf-8'))
    w = load_corpus('chorales', VOCAB, win, 8)
    _, vl, val_ids = group_split(w, 0.08, 42)
    pool = [p for p in pieces if p['id'] in val_ids and len(p['slices']) >= win][:n]
    out = []
    for i, p in enumerate(pool):
        ww = piece_to_window(p, VOCAB, 0, win, 0)
        T = ww['pitch'].shape[1]
        cond = {k: ww[k][None].to(device) for k in COND_KEYS}
        pitch = torch.full((1, 1, T), VOCAB.rest, dtype=torch.long, device=device)
        rhythm = torch.full((1, 1, T), HOLD, dtype=torch.long, device=device)
        known = torch.zeros((1, 1, T), dtype=torch.bool, device=device)
        torch.manual_seed(seed + i)
        gp, gr = model.generate(pitch, rhythm, known, cond, steps=1, argmax=False, temp=temp)
        # 乐句末切片（用于指标）
        ends = [ph['end_slice'] for ph in p['phrases'] if ph['end_slice'] < T]
        out.append((gp[0, 0].cpu(), gr[0, 0].cpu(), ends))
    return out


@torch.no_grad()
def gen_melodies(ckpt, d, layers, n, bars, seed, device, temp=0.9):
    model = SATBDiffusion(d=d, h=8, layers=layers, vocab=VOCAB, n_voices=1).to(device)
    model.load_state_dict(torch.load(MD / ckpt, map_location=device, weights_only=True))
    model.eval()
    T = bars * 4
    out = []
    for i in range(n):
        cond, ends = build_plan(T, VOCAB, device)
        pitch = torch.full((1, 1, T), VOCAB.rest, dtype=torch.long, device=device)
        rhythm = torch.full((1, 1, T), HOLD, dtype=torch.long, device=device)
        known = torch.zeros((1, 1, T), dtype=torch.bool, device=device)
        torch.manual_seed(seed + i)
        gp, gr = model.generate(pitch, rhythm, known,
                                {k: cond[k] for k in COND_KEYS},
                                steps=1, argmax=False, temp=temp)
        out.append((gp[0, 0].cpu(), gr[0, 0].cpu(), ends))
    return out


@torch.no_grad()
def recovery_metrics(model, harmony_unknown: bool, device, win=64, stride=8) -> dict:
    """掩码恢复准确率（有真值、可复现），与音乐性指标一起落盘。

    口径必须写清：**和声条件是否喂真值**由 `harmony_unknown` 决定 ——
    训练时没有和声条件的模型（v13）必须置"未知"，否则喂的是它没见过的条件，
    数字不可比（实测同一检查点在两种条件下差 0.2–0.6）。
    """
    from torch.utils.data import DataLoader

    from train_v10_satb import WindowDataset, eval_recovery
    w = load_corpus('chorales', VOCAB, win, stride)
    _, vl, _ = group_split(w, 0.08, 42)
    rec = eval_recovery(model, DataLoader(WindowDataset(vl), batch_size=16), device,
                        harmony_unknown=harmony_unknown)
    return {'mask_recovery': rec,
            'mask_recovery_cond': '和声条件=未知' if harmony_unknown else '和声条件=真值',
            'mask_recovery_win': win}


def describe(gens) -> dict:
    """生成旋律的同一批音乐性指标（口径与 corpus_baseline 一致）。"""
    all_iv = []          # 全部相邻音程
    steps = 0
    dirchg = []
    ranges, dens, phr_stable = [], [], []
    last_tonic = 0
    for pcs_t, rhy_t, ends in gens:
        pcs, durs = [], []
        for t in range(len(pcs_t)):
            p = int(pcs_t[t])
            if p >= VOCAB.n_pitch:
                continue
            pcs.append(p + VOCAB.lo)
            if int(rhy_t[t]) < HOLD:
                durs.append(RHYTHM_VALUES[min(int(rhy_t[t]), 7)])
        if len(pcs) < 8:
            continue
        iv = [abs(b - a) for a, b in zip(pcs[:-1], pcs[1:])]
        all_iv += iv
        steps += sum(1 for x in iv if x <= 2)
        ranges.append(max(pcs) - min(pcs))
        dens.append(len(durs) / max(sum(durs), 1e-6))
        if pcs[-1] % 12 == 0:
            last_tonic += 1
        for e in ends:
            if e < len(pcs_t) and int(pcs_t[e]) < VOCAB.n_pitch:
                phr_stable.append((int(pcs_t[e]) + VOCAB.lo) % 12 in (0, 7))
        signs = [1 if b > a else -1 for a, b in zip(pcs[:-1], pcs[1:]) if b != a]
        if len(signs) > 1:
            dirchg.append(sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b) / (len(signs) - 1))
    n_ok = sum(1 for g in gens if sum(1 for x in g[0].tolist() if x < VOCAB.n_pitch) >= 8)
    return {
        'step_ratio': steps / max(len(all_iv), 1),
        'avg_interval': statistics.mean(all_iv) if all_iv else 0,
        'direction_change': statistics.mean(dirchg) if dirchg else 0,
        'range_span': statistics.mean(ranges) if ranges else 0,
        'onset_density': statistics.mean(dens) if dens else 0,
        'last_note_tonic': last_tonic / max(n_ok, 1),
        'phrase_end_stable': (sum(phr_stable) / len(phr_stable)) if phr_stable else 0,
        'n_generated': len(gens),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='v14_melody.pt')
    ap.add_argument('--melody-d', type=int, default=384)
    ap.add_argument('--melody-layers', type=int, default=6)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--bars', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temp', type=float, default=0.9)
    ap.add_argument('--out-dir', type=str, default='data/generated/melody')
    ap.add_argument('--ref', type=str, default=None, help='另一个检查点（A/B 对照）')
    ap.add_argument('--scored', action='store_true', help='用乐理评分引导旋律解码')
    ap.add_argument('--with-harmony', action='store_true',
                    help='诊断: 用真值曲子的和声条件生成旋律（检验"和声锚定"假设）')
    ap.add_argument('--out', type=str, default='data/processed/melody_eval.json')
    ap.add_argument('--recovery-harmony', choices=['auto', 'unknown', 'true'], default='auto',
                    help='掩码恢复评估时的和声条件；auto = 有 --with-harmony 则真值, 否则未知')
    ap.add_argument('--no-recovery', action='store_true', help='跳过掩码恢复评估')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ref = corpus_baseline()
    print(f'真众赞歌 Soprano 基线 ({ref["n_pieces"]} 首): 级进率 {ref["step_ratio"]:.3f} | '
          f'平均音程 {ref["avg_interval"]:.2f} | 方向变化 {ref["direction_change"]:.2f} | '
          f'音域跨度 {ref["range_span"]:.1f} | 乐句末稳定音 {ref["phrase_end_stable"]:.2f} | '
          f'末音主音 {ref["last_note_tonic"]:.2f}')
    out = {'corpus': ref}
    for tag, ck in ((Path(args.ckpt).stem, args.ckpt),) + (
            ((Path(args.ref).stem, args.ref),) if args.ref else ()):
        if args.with_harmony:
            model = SATBDiffusion(d=args.melody_d, h=8, layers=args.melody_layers,
                                  vocab=VOCAB, n_voices=1).to(device)
            model.load_state_dict(torch.load(MD / ck, map_location=device, weights_only=True))
            model.eval()
            gens = gen_melodies_with_harmony(model, args.n, args.seed, device, args.temp)
        elif args.scored:
            model = SATBDiffusion(d=args.melody_d, h=8, layers=args.melody_layers,
                                  vocab=VOCAB, n_voices=1).to(device)
            model.load_state_dict(torch.load(MD / ck, map_location=device, weights_only=True))
            model.eval()
            gens = []
            T = args.bars * 4
            for i in range(args.n):
                cond, ends = build_plan(T, VOCAB, device)
                gp, gr = gen_melody_scored(model, cond, T, device, seed=args.seed + i,
                                           temp=args.temp)
                gens.append((gp[0], gr, ends))          # 两者都已是 [T]
        else:
            gens = gen_melodies(ck, args.melody_d, args.melody_layers, args.n, args.bars,
                                args.seed, device, args.temp)
        r = describe(gens)
        if not args.no_recovery:
            m2 = SATBDiffusion(d=args.melody_d, h=8, layers=args.melody_layers,
                               vocab=VOCAB, n_voices=1).to(device)
            m2.load_state_dict(torch.load(MD / ck, map_location=device, weights_only=True))
            m2.eval()
            hu = (args.recovery_harmony == 'unknown' or
                  (args.recovery_harmony == 'auto' and not args.with_harmony))
            r.update(recovery_metrics(m2, hu, device))
        out[tag] = r
        print(f'\n{tag} ({ck}, {r["n_generated"]} 首): 级进率 {r["step_ratio"]:.3f} | '
              f'平均音程 {r["avg_interval"]:.2f} | 方向变化 {r["direction_change"]:.2f} | '
              f'音域跨度 {r["range_span"]:.1f} | 乐句末稳定音 {r["phrase_end_stable"]:.2f} | '
              f'末音主音 {r["last_note_tonic"]:.2f}')
        if 'mask_recovery' in r:
            print(f'  掩码恢复（{r["mask_recovery_cond"]}, win{r["mask_recovery_win"]}）: '
                  + ' '.join(f'{k.replace("rec_pitch_", "")}={v:.3f}'
                             for k, v in r['mask_recovery'].items()
                             if k.startswith('rec_pitch')))
        od = MD / args.out_dir / tag.split('(')[0]
        od.mkdir(parents=True, exist_ok=True)
        for i, (p, rh, _) in enumerate(gens[:3]):
            write_midi(od / f'mel_{i + 1}.mid', torch.stack([p]), torch.stack([rh]),
                       [float(t) for t in range(len(p))], VOCAB, 76)
    out['_config'] = {**vars(args), 'device': device}
    json.dump(out, open(MD / args.out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'\n→ {args.out} 与 data/generated/melody/')


if __name__ == '__main__':
    main()
