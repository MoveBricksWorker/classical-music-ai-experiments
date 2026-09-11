"""
v9 句子级训练 —— 修复验证集泄漏 (按曲分组划分) + 边界头类别加权 + 多种子报告。

与 v8 的差异**只在实验方法, 不在模型结构**:
    1. 划分: 按曲分组 (GroupShuffleSplit / k 折)。v8 用样本级随机划分,
       而滑窗 win=32/stride=6 使同曲相邻窗口重叠 81% → 96% 的验证窗口
       与训练窗口同曲重叠, 恢复准确率与边界 F1 被系统性高估;
    2. 边界头: 句末音仅占 ~8.5%, 原始 BCE 使模型倾向全判负 (召回 0.05);
       加 pos_weight (默认 5.0);
    3. 评估: 恢复准确率按多个掩码比例; 边界 F1 支持阈值扫描 / ±容差 /
       按终止式类型分层; k 折报告 均值±标准差 (单次划分的 0.08-0.35
       波动本身不可解释)。

用法:
    python train_v9_chorales.py --folds 5 --epochs 120 --tag v9            # 交叉验证
    python train_v9_chorales.py --folds 1 --val-ratio 0.1 --tag v9_main    # 主模型
    python train_v9_chorales.py --folds 1 --split sample --tag v9_leaked   # A/B: 旧泄漏划分
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import argparse, json, random, time, statistics
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

from chorale_conditions import (load_chorale_windows, grouped_split,
                                kfold_by_piece, CADENCE_NAME)
from constants import F2ID_MELODY, T2ID
from model.melody_diffusion import MelodyDiffusion, PITCH_MASK, RHYTHM_MASK, REST
from train_v8_chorales import ChoraleDataset, batch_to_dev

DATA = str(ROOT / 'data/processed/chorales_sentences_v1.json')
FIELDS = ['func', 'type', 'root', 'pc', 'rhythm', 'pos_bin', 'toend_bin',
          'phrase_bin', 'cadence', 'boundary']


def build_model(args):
    return MelodyDiffusion(d=args.d, h=8, L=args.layers, max_len=256,
                           num_funcs=len(F2ID_MELODY), num_types=len(T2ID),
                           num_pos_bins=8, num_toend_bins=4, num_phrase_bins=7,
                           num_cadence_types=5, use_rope=True)


# ─────────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_recovery(model, loader, device, ratios=(0.15, 0.5, 0.85), n_batches=8):
    """掩码恢复准确率 (音高), 按多个掩码比例报告 —— 单一 t 的数值不可比。"""
    model.eval()
    out = {}
    for t in ratios:
        cor = tot = 0
        for bi, b in enumerate(loader):
            if bi >= n_batches:
                break
            fn, tp, rt, pc, rh, pb, tb, phb, cad, bd = batch_to_dev(b, device)
            B, L = pc.shape
            mask = torch.rand(B, L, device=device) < t
            p_in = pc.clone(); r_in = rh.clone()
            p_in[mask] = PITCH_MASK; r_in[mask] = RHYTHM_MASK
            flag = (~mask).long() * 2 + (~mask).long()
            pl, _, _ = model(p_in, r_in, fn, tp, rt, flag, None, pb, tb, phb, cad,
                             return_boundary=True)
            cor += (pl.argmax(-1)[mask] == pc[mask]).sum().item()
            tot += mask.sum().item()
        out[f'recovery_t{int(t*100)}'] = cor / max(tot, 1)
    return out


@torch.no_grad()
def eval_boundary(model, loader, device):
    """边界头评估: 遮蔽结构流, 从旋律+和声内容判断句末。

    返回 pooled P/R/F1 (阈值 0.5, 无容差) + 阈值扫描 + ±1/±2 容差 F1 +
    按终止式类型分层的召回。
    """
    model.eval()
    all_pred, all_true, all_cad = [], [], []
    for b in loader:
        fn, tp, rt, pc, rh, pb, tb, phb, cad, bd = batch_to_dev(b, device)
        flag = torch.ones_like(pc)
        phb = torch.full_like(phb, 6)          # 结构流遮蔽 (防泄漏)
        cad_in = torch.zeros_like(cad)
        _, _, bl = model(pc, rh, fn, tp, rt, flag, None, pb, tb, phb, cad_in,
                         return_boundary=True)
        prob = torch.softmax(bl, -1)[..., 1]
        all_pred.append(prob.cpu())
        all_true.append(bd.cpu())
        all_cad.append(cad.cpu())              # 真值终止式类型 (仅用于分层)
    prob = torch.cat(all_pred); true = torch.cat(all_true); cad = torch.cat(all_cad)

    def prf(pred, tru):
        tp_ = ((pred == 1) & (tru == 1)).sum().item()
        fp_ = ((pred == 1) & (tru == 0)).sum().item()
        fn_ = ((pred == 0) & (tru == 1)).sum().item()
        p = tp_ / max(tp_ + fp_, 1); r = tp_ / max(tp_ + fn_, 1)
        return 2 * p * r / max(p + r, 1e-6), p, r

    res = {}
    # 阈值扫描 (单点口径)
    sweep = {}
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        sweep[str(th)] = prf((prob > th).long(), true)[0]
    res['f1_sweep'] = sweep
    res['best_threshold'] = max(sweep, key=sweep.get)
    res['best_f1'] = sweep[res['best_threshold']]
    f1_05, p_05, r_05 = prf((prob > 0.5).long(), true)
    res['f1_at_0.5'] = f1_05
    res['precision'] = p_05
    res['recall'] = r_05

    # ±容差 F1 (贪心一对一匹配, 逐窗口)
    for tol in (1, 2):
        tp_ = fp_ = fn_ = 0
        for row_p, row_t in zip((prob > 0.5), true):
            preds = row_p.nonzero().flatten().tolist()
            trues = row_t.nonzero().flatten().tolist()
            used = set(); m = 0
            for p_ in preds:
                for t_ in trues:
                    if t_ in used:
                        continue
                    if abs(p_ - t_) <= tol:
                        m += 1; used.add(t_); break
            tp_ += m; fp_ += len(preds) - m; fn_ += len(trues) - m
        pp = tp_ / max(tp_ + fp_, 1); rr = tp_ / max(tp_ + fn_, 1)
        res[f'f1_tol{tol}'] = 2 * pp * rr / max(pp + rr, 1e-6)

    # 按终止式类型分层的召回 (只统计非"弱分句"的真值边界)
    cad_names = {1: 'authentic', 2: 'half', 3: 'deceptive', 4: 'plagal'}
    per_cad = {}
    for cid, cname in cad_names.items():
        m = (true == 1) & (cad == cid)
        if m.sum() == 0:
            continue
        rec = ((prob > 0.5).long()[m] == 1).float().mean().item()
        per_cad[cname] = {'n': int(m.sum().item()), 'recall': rec}
    res['recall_by_cadence'] = per_cad
    return res


@torch.no_grad()
def eval_generation(model, samples, device, n=48, steps=16, seed=0):
    """全掩码生成 (条件含真实乐句/终止式计划), 与真值对比音高准确率。"""
    model.eval()
    rng = random.Random(seed)
    idxs = rng.sample(range(len(samples)), min(n, len(samples)))
    cor = tot = 0
    for i in idxs:
        s = samples[i]
        fn = torch.tensor([s['func']], device=device)
        tp = torch.tensor([s['type']], device=device)
        rt = torch.tensor([s['root']], device=device)
        pb = torch.tensor([s['pos_bin']], device=device)
        tb = torch.tensor([s['toend_bin']], device=device)
        phb = torch.tensor([s['phrase_bin']], device=device)
        cad = torch.tensor([s['cadence']], device=device)
        gp, _ = model.generate(fn, tp, rt, steps=steps, temp=1.0, expand=False,
                               pos_bin=pb, toend_bin=tb, phrase_bin=phb, cadence=cad,
                               seed=rng.randint(0, 10 ** 6))
        gt = torch.tensor(s['pc'], device=device)
        cor += (gp[0] == gt).sum().item()
        tot += len(s['pc'])
    return cor / max(tot, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 单次训练
# ─────────────────────────────────────────────────────────────────────────────

def train_one(tr_samples, vl_samples, args, seed, out_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    model = build_model(args).to(device)
    tr = DataLoader(ChoraleDataset(tr_samples), batch_size=args.batch, shuffle=True)
    vl = DataLoader(ChoraleDataset(vl_samples), batch_size=args.batch)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    t_all = time.time()
    for ep in range(args.epochs):
        model.train()
        tl = nb = 0
        for b in tr:
            fn, tp_, rt, pc, rh, pb, tb, phb, cad, bd = batch_to_dev(b, device)
            if random.random() < args.struct_drop:
                phb = torch.full_like(phb, 6)
                cad = torch.zeros_like(cad)
            opt.zero_grad()
            loss, _, _, _ = model.training_loss(
                pc, rh, fn, tp_, rt, beat=None, pos_bin=pb, toend_bin=tb,
                phrase_bin=phb, cadence=cad, boundary=bd,
                boundary_weight=args.bw, boundary_pos_weight=args.bw_pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); nb += 1
        sched.step()

        if (ep + 1) % max(1, args.epochs // 8) == 0 or ep == args.epochs - 1:
            model.eval()
            brep = eval_boundary(model, vl, device)
            rec = eval_recovery(model, vl, device, ratios=(0.5,))['recovery_t50']
            gen = eval_generation(model, vl_samples, device, n=32, seed=seed)
            # 选型分数: 边界 F1 (±1 容差) + 生成准确率 (与 v8 口径一致但用容差)
            score = brep['f1_tol1'] + 0.3 * gen
            if score > best:
                best = score
                torch.save(model.state_dict(), str(out_path))
            print(f'  Ep{ep+1:3d} | rec:{rec:.3f} | F1@.5:{brep["f1_at_0.5"]:.3f} '
                  f'(P{brep["precision"]:.2f}/R{brep["recall"]:.2f}) | F1±1:{brep["f1_tol1"]:.3f} '
                  f'| gen:{gen:.3f} | {time.time()-t_all:.0f}s', flush=True)

    model.load_state_dict(torch.load(str(out_path), map_location=device, weights_only=True))
    model.eval()
    m = {'seed': seed, 'n_train': len(tr_samples), 'n_val': len(vl_samples)}
    m.update(eval_recovery(model, vl, device))
    m.update(eval_boundary(model, vl, device))
    m['generation_acc'] = eval_generation(model, vl_samples, device, n=48, seed=seed)
    m['train_seconds'] = round(time.time() - t_all, 1)
    return model, m


def _col(name, per_fold):
    vals = [f[name] for f in per_fold if isinstance(f.get(name), (int, float))]
    if not vals:
        return None
    return {'mean': round(statistics.mean(vals), 4),
            'std': round(statistics.pstdev(vals), 4), 'values': [round(v, 4) for v in vals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--d', type=int, default=320)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--win', type=int, default=32)
    ap.add_argument('--stride', type=int, default=6)
    ap.add_argument('--bw', type=float, default=0.5, help='边界损失权重 (v8=0.3)')
    ap.add_argument('--bw-pos-weight', type=float, default=5.0,
                    help='边界头正类权重 (正类仅 ~8.5%)')
    ap.add_argument('--struct-drop', type=float, default=0.5)
    ap.add_argument('--split', choices=['group', 'sample'], default='group')
    ap.add_argument('--folds', type=int, default=1, help='>1 时按曲 k 折交叉验证')
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', type=str, default='v9')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    samples = load_chorale_windows(DATA, args.win, args.stride)
    print(f'设备 {device} | 窗口 {len(samples)} | 曲 {len({s["piece"] for s in samples})}')

    if args.split == 'group' and args.folds > 1:
        folds = kfold_by_piece(samples, k=args.folds, seed=args.seed)
        print(f'按曲 {args.folds} 折交叉验证 (每折 val {len(folds[0][1])} 窗口 / '
              f'{len(folds[0][2])} 首)')
    elif args.split == 'group':
        tr, vl, vp = grouped_split(samples, args.val_ratio, args.seed)
        folds = [(tr, vl, vp)]
        print(f'按曲分组划分: train {len(tr)} / val {len(vl)} ({len(vp)} 首)')
    else:
        rng = random.Random(args.seed)
        idx = list(range(len(samples))); rng.shuffle(idx)
        sp = int(len(idx) * (1 - args.val_ratio))
        folds = [([samples[i] for i in idx[:sp]], [samples[i] for i in idx[sp:]], set())]
        print(f'样本级随机划分 (v8 旧口径, 有泄漏): train {sp} / val {len(idx)-sp}')

    per_fold, reports = [], []
    for k, (tr, vl, vp) in enumerate(folds):
        out = ROOT / (args.out or (f'melody_diffusion_{args.tag}.pt' if len(folds) == 1
                                   else f'melody_diffusion_{args.tag}_fold{k}.pt'))
        print(f'── fold {k}: train {len(tr)} / val {len(vl)} → {out.name}')
        _, m = train_one(tr, vl, args, seed=args.seed + k, out_path=out)
        m['fold'] = k
        per_fold.append(m)
        reports.append(m)
        print(f'   fold{k}: rec(t50) {m["recovery_t50"]:.3f} | F1@.5 {m["f1_at_0.5"]:.3f} '
              f'| F1±1 {m["f1_tol1"]:.3f} | F1±2 {m["f1_tol2"]:.3f} '
              f'| P {m["precision"]:.2f} R {m["recall"]:.2f} | gen {m["generation_acc"]:.3f}')

    summary = {
        'config': vars(args),
        'n_folds': len(folds),
        'recovery_t15': _col('recovery_t15', per_fold),
        'recovery_t50': _col('recovery_t50', per_fold),
        'recovery_t85': _col('recovery_t85', per_fold),
        'boundary_f1_at_0.5': _col('f1_at_0.5', per_fold),
        'boundary_f1_tol1': _col('f1_tol1', per_fold),
        'boundary_f1_tol2': _col('f1_tol2', per_fold),
        'boundary_precision': _col('precision', per_fold),
        'boundary_recall': _col('recall', per_fold),
        'generation_acc': _col('generation_acc', per_fold),
        'per_fold': per_fold,
    }
    out_json = ROOT / f'data/processed/{args.tag}_report.json'
    json.dump(summary, open(out_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n汇总 → {out_json}')
    for key in ('recovery_t50', 'boundary_f1_at_0.5', 'boundary_f1_tol1',
                'boundary_f1_tol2', 'boundary_precision', 'boundary_recall',
                'generation_acc'):
        c = summary[key]
        if c:
            print(f'  {key:24s} {c["mean"]:.3f} ± {c["std"]:.3f}')


if __name__ == '__main__':
    main()
