"""
"规划器和声 vs 真和声标签 vs 无和声"三条件对照 —— 给报告里的三级流水线里程碑补证据链。

`data/processed/v13_planned.json` 此前**没有任何脚本能生成它**（报告 08 §3 的表格
数字与产物本身也对不上），这个脚本就是为了让那张表可复现、可审计。

任务固定为"给定真旋律（Soprano）+ 某种和声条件 → 写内声部"，三种条件：

    true    真值和声标签（func/type/root 来自语料，旧口径的上界）
    planned 和声规划器**生成**的和声（自回归，以旋律+曲式模板为输入）
    none    和声条件标为"未知"（模型必须自己定和声）

指标（同一批留出曲、同一解码 `steps=1, argmax=True`，与 `eval_harmonize` 同口径）：

    acc                内声部逐音命中真值的比例（**只在真值为实音的位置统计**）
    chord_explainable  纵向音响（含已知的 Soprano）能被某个已知和弦解释的切片比例

注意 `acc` 在"生成"任务里不是主指标：规划器生成的是另一条合理的和声，
不是巴赫那一条 —— 要看的是它有没有掉到"无和声"那一档。

用法
----
    python analyze_planned_harmony.py --n 48 \
        --realizer v11_scratch.pt --planner harmony_planner_mel.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import torch

from gen_from_scratch import VOCAB, chord_explainable, corpus_reference
from gen_full_pipeline import plan_harmony
from model.satb_diffusion import N_VOICES, SATBDiffusion
from train_harmony_planner import HarmonyPlanner
from train_v10_satb import HOLD, load_corpus, group_split

CONDS = ('true', 'planned', 'none')


@torch.no_grad()
def eval_condition(realizer, windows, device, cond_mode, planner=None, seed=0):
    """在留出窗上跑一种和声条件 → {acc, chord_explainable, n_windows}。"""
    hit = tot = chord_ok = chord_tot = 0
    for i, w in enumerate(windows):
        pitch = w['pitch'][None].to(device)
        rhythm = w['rhythm'][None].to(device)
        cond = {k: w[k][None].to(device) for k in
                ('func', 'type', 'root', 'pos_bin', 'toend_bin', 'phrase_bin', 'cadence', 'style')}
        if cond_mode == 'planned':
            cond = plan_harmony(planner, cond, pitch[0, 0:1], pitch.shape[2], device,
                               seed=seed + i)          # sop 需为 [1, T]
        elif cond_mode == 'none':
            for k, unk in (('func', 7), ('type', 9), ('root', 12)):
                cond[k] = torch.full_like(cond[k], unk)
        known = torch.zeros_like(pitch, dtype=torch.bool)
        known[:, 0] = True                                  # 只给 Soprano
        gp, _ = realizer.generate(pitch, rhythm, known, cond, steps=1, argmax=True)
        tgt = pitch[0]
        for v in range(1, N_VOICES):                        # 内声部逐音命中（真值为实音）
            real = tgt[v] < realizer.vocab.n_pitch
            hit += int(((gp[0, v] == tgt[v]) & real).sum().item())
            tot += int(real.sum().item())
        for t in range(pitch.shape[2]):                     # 纵向音响可解释为和弦
            ps = [int(gp[0, v, t]) for v in range(N_VOICES)
                  if int(gp[0, v, t]) < realizer.vocab.n_pitch]
            if not ps:
                continue
            chord_tot += 1
            chord_ok += int(chord_explainable([(p + realizer.vocab.lo) % 12 for p in ps]))
    return {'acc': hit / max(tot, 1), 'chord_explainable': chord_ok / max(chord_tot, 1),
            'n_windows': len(windows), 'n_notes_scored': tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--realizer', type=str, default='v11_scratch.pt')
    ap.add_argument('--realizer-d', type=int, default=480)
    ap.add_argument('--realizer-layers', type=int, default=8)
    ap.add_argument('--planner', type=str, default='harmony_planner_mel.pt')
    ap.add_argument('--planner-d', type=int, default=256)
    ap.add_argument('--planner-layers', type=int, default=4)
    ap.add_argument('--n', type=int, default=48, help='留出窗数（与 eval_harmonize 同口径）')
    ap.add_argument('--win', type=int, default=64)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='data/processed/v13_planned.json')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    windows = load_corpus('chorales', VOCAB, args.win, args.stride)
    _, vl, val_ids = group_split(windows, 0.08, 42)
    sample = vl if len(vl) <= args.n else __import__('random').Random(args.seed).sample(vl, args.n)

    realizer = SATBDiffusion(d=args.realizer_d, h=8, layers=args.realizer_layers,
                             vocab=VOCAB).to(device)
    realizer.load_state_dict(torch.load(ROOT / args.realizer, map_location=device,
                                        weights_only=True))
    realizer.eval()
    planner = None
    if 'planned' in CONDS:
        planner = HarmonyPlanner(d=args.planner_d, layers=args.planner_layers).to(device)
        planner.load_state_dict(torch.load(ROOT / args.planner, map_location=device,
                                          weights_only=True))
        planner.eval()

    print(f'留出曲 {len(val_ids)} 首 / 评估窗口 {len(sample)} | 实现器 {args.realizer} '
          f'| 规划器 {args.planner}')
    out = {'config': vars(args), 'n_val_pieces': len(val_ids)}
    for mode in CONDS:
        r = eval_condition(realizer, sample, device, mode, planner, seed=args.seed)
        out[mode] = r
        print(f'  {mode:8s} 内声部逐音命中 {r["acc"]:.3f} | 纵向和弦率 '
              f'{r["chord_explainable"]:.3f}（{r["n_notes_scored"]} 音）')
    ref = corpus_reference()
    out['reference'] = {k: v for k, v in ref.items() if k != 'key_dist'}
    print(f'  真巴赫   纵向和弦率 {ref["chord_explainable"]:.3f} | C 调占比 '
          f'{ref["key_c_ratio"]:.3f}')
    json.dump(out, open(ROOT / args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
