"""
泄漏分解实验 —— 把"同曲窗口重叠"与"训练曲目更多"两个因素分开。

背景: v8 的样本级随机划分有两个后果 ——
  (a) 验证窗口与训练窗口同曲、81% 时间重叠;
  (b) 训练集覆盖 363/368 首 (几乎全部曲目), 而按曲分组只覆盖 331 首。
A/B 对照 (train_v9_chorales --split sample vs group) 显示旧口径的恢复率
约高一倍, 但两者混淆了 (a) 与 (b)。本实验固定训练集为同一批**曲**,
只改变验证窗口是否与训练窗口重叠:

    训练: 每首的 start%12==0 窗口;  验证A: 同曲的 start%12==6 窗口
                                   验证B: 完全没见过的曲的全部窗口

用法: python analyze_leakage.py --epochs 120
输出: data/processed/leak_decomposition.json
"""
import sys, json, random, argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))

import torch
from torch.utils.data import DataLoader

from chorale_conditions import load_chorale_windows, grouped_split
from train_v8_chorales import ChoraleDataset
from train_v9_chorales import train_one, eval_recovery, eval_boundary, eval_generation

DATA = str(ROOT / 'data/processed/chorales_sentences_v1.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--bw', type=float, default=0.5)
    ap.add_argument('--bw-pos-weight', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    samples = load_chorale_windows(DATA)              # win 32 / stride 6
    tr_pool, unseen_pieces, vp = grouped_split(samples, 0.1, 42)

    # 同一批曲 (tr_pool 的曲), 按窗口起点下标对半切: 偶数训练 / 奇数留出
    train = [s for s in tr_pool if s['start'] % 12 == 0]
    held_same_piece = [s for s in tr_pool if s['start'] % 12 == 6]
    print(f'训练 {len(train)} 窗口 ({len({s["piece"] for s in train})} 首) | '
          f'同曲留出 {len(held_same_piece)} | 未见曲 {len(unseen_pieces)} ({len(vp)} 首)')

    ns = argparse.Namespace(epochs=args.epochs, batch=64, lr=2e-4, d=320, layers=6,
                            win=32, stride=6, bw=args.bw,
                            bw_pos_weight=args.bw_pos_weight, struct_drop=0.5)
    out_pt = ROOT / 'melody_diffusion_leakcheck.pt'
    model, m_train = train_one(train, held_same_piece, ns, seed=args.seed, out_path=out_pt)

    res = {'n_train': len(train), 'n_held_same_piece': len(held_same_piece),
           'n_unseen_piece': len(unseen_pieces)}
    for tag, val in (('held_same_piece', held_same_piece), ('unseen_piece', unseen_pieces)):
        ld = DataLoader(ChoraleDataset(val), batch_size=64)
        r = eval_recovery(model, ld, device)
        b = eval_boundary(model, ld, device)
        g = eval_generation(model, val, device, n=48, seed=args.seed)
        res[tag] = {**r, **{f'b_{k}': v for k, v in b.items() if not isinstance(v, dict)},
                    'generation_acc': g}
        print(f'{tag:18s} rec t.50 {r["recovery_t50"]:.3f} | F1@.5 {b["f1_at_0.5"]:.3f} '
              f'(P{b["precision"]:.2f}/R{b["recall"]:.2f}) | F1±1 {b["f1_tol1"]:.3f} | gen {g:.3f}')

    # 把"同曲未训窗口"与"未见曲"的差直接算出来
    res['inflation'] = {
        k: round(res['held_same_piece'][k] - res['unseen_piece'][k], 4)
        for k in ('recovery_t15', 'recovery_t50', 'recovery_t85', 'generation_acc')}
    print('膨胀量 (同曲未训窗口 - 未见曲):', res['inflation'])
    out = ROOT / (args.out or 'data/processed/leak_decomposition.json')
    json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'→ {out}')


if __name__ == '__main__':
    main()
