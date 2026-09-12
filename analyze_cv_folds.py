"""
严格三折交叉验证的**汇总评估** —— 把 `v10_cv_fold{k}.pt` 在三折留出曲上复评并落盘。

为什么单独写一个：
1. 训练时的评估只抽 `--eval-n`（默认 48）个窗口，而每折留出曲有 1000+ 窗口，
   48 个样本的估计噪声太大；这里用大样本（默认 400 窗）复评同一个检查点；
2. 训练日志只在终端，产物里要有一份可直接引用的 mean±std；
3. 评估必须用**未增广**的窗口（音乐按原样，全在 C 调）——
   增广只用于训练，拿 5 个移调副本去评估会把同一段音乐数 5 遍。

口径：按曲 3 折（`kfold_by_piece(seed=42)`），每折模型只在该折留出曲上评估；
解码 `steps=1, argmax=True`；和声条件 = 真值（与"配和声"任务一致）。

⚠️ 两种验证集口径**差一倍**，必须分别报（2026-09-12 实测，同一批检查点）：

    main（标准）  未增广：音乐按原样（都移调到 C）   折 0.849/0.571/0.699 → 0.706±0.114
    aug（训练日志）增广  ：混入 ±2 移调副本，等于要求"在别的调里也能配"
                                                     折 0.454/0.440/0.381 → 0.425±0.032

训练时打印的就是后者（`build_split(args.aug_transpose)` 拿增广窗口当验证集），
所以**训练日志的 0.43–0.48 不能与报告 07 的 0.877 直接比**（后者走
`--eval-only`，用未增广窗口）。默认两次都跑，产物里分别放在 `main` / `aug`。

用法
----
    python analyze_cv_folds.py --folds 3 --eval-n 400          # 两种口径都评
    python analyze_cv_folds.py --folds 3 --no-aug-eval         # 只评标准口径
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import torch
from torch.utils.data import DataLoader

from chorale_conditions import kfold_by_piece
from model.satb_diffusion import SATBDiffusion, SatbVocab
from train_v10_satb import (WindowDataset, eval_harmonize, eval_recovery,
                            load_corpus)

METRIC_KEYS = ('harm_pitch_acc_masked', 'harm_pitch_acc_chorale', 'harm_rhythm_acc',
               'parallel_5_per100', 'ref_parallel_5_per100', 'parallel_8_per100',
               'ref_parallel_8_per100', 'crossing_per100', 'ref_crossing_per100',
               'spacing_per100', 'ref_spacing_per100')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', type=str, default='v10_cv_fold{}.pt')
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--eval-n', type=int, default=400, help='每折评估窗口数')
    ap.add_argument('--d', type=int, default=480)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--win', type=int, default=64)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42, help='k 折划分的种子（与训练一致）')
    ap.add_argument('--aug-transpose', type=int, default=2, help='训练用的移调增广幅度')
    ap.add_argument('--no-aug-eval', action='store_true', help='跳过增广验证集口径')
    ap.add_argument('--out', type=str, default='data/processed/v10_cv_summary.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = SatbVocab()

    def eval_all(tag: str, transpose_aug: int) -> dict:
        """在给定验证集口径上评估三折 → {folds, mean_std}。"""
        windows = load_corpus('chorales', vocab, args.win, args.stride,
                              transpose_aug=transpose_aug)
        folds = kfold_by_piece(windows, k=args.folds, seed=args.seed)
        per_fold = []
        for k, (_, vl, val_ids) in enumerate(folds):
            ckpt = ROOT / args.pattern.format(k)
            if not ckpt.exists():
                print(f'跳过第 {k} 折: 缺少 {ckpt.name}')
                continue
            model = SATBDiffusion(d=args.d, h=args.heads, layers=args.layers,
                                  vocab=vocab).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.eval()
            rec = eval_recovery(model, DataLoader(WindowDataset(vl), batch_size=16), device)
            n = args.eval_n if tag == 'main' else min(args.eval_n, 48)
            harm = eval_harmonize(model, vl, device, n=n)
            row = {'fold': k, 'ckpt': ckpt.name, 'n_val_pieces': len(val_ids),
                   'n_val_windows': len(vl),
                   **{key: harm[key] for key in METRIC_KEYS if key in harm},
                   **{f'rec_{key}': v for key, v in rec.items()},
                   'per_voice_pitch_acc': harm.get('per_voice_pitch_acc')}
            per_fold.append(row)
            print(f'[{tag}] 折 {k} ({len(val_ids)} 首 / {len(vl)} 窗, 评估 '
                  f'{harm.get("n_eval_windows")}): 配和声音高 {row["harm_pitch_acc_masked"]:.3f} | '
                  f'节奏 {row["harm_rhythm_acc"]:.3f} | 平行五 {row["parallel_5_per100"]:.2f} '
                  f'vs 真值 {row["ref_parallel_5_per100"]:.2f}')
        keys = [k for k in METRIC_KEYS if per_fold and k in per_fold[0]]
        return {'transpose_aug': transpose_aug, 'folds': per_fold,
                'mean_std': {k: {'mean': statistics.mean([f[k] for f in per_fold]),
                                 'std': statistics.pstdev([f[k] for f in per_fold])}
                             for k in keys}}

    summary = {'config': vars(args), 'n_folds': args.folds,
               'main': eval_all('main', 0)}
    if not args.no_aug_eval:
        summary['aug'] = eval_all('aug', args.aug_transpose)
    json.dump(summary, open(ROOT / args.out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'\n{args.folds} 折汇总（每折留出曲完全没参与训练）→ {args.out}')
    for tag in ('main', 'aug'):
        if tag not in summary:
            continue
        c = summary[tag]['mean_std']['harm_pitch_acc_masked']
        label = '未增广（标准）' if tag == 'main' else '增广（训练日志口径）'
        print(f'  {label:20s} 配和声音高 {c["mean"]:.3f} ± {c["std"]:.3f}')


if __name__ == '__main__':
    main()
