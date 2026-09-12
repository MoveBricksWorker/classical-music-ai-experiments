"""
解码策略实验：多候选 + 乐理评分重排（theory-guided decoding, v10 版）。

背景
----
v10 首版（单步 argmax）配和声音高准确率 0.554，但**平行五度 0.573/100 对**，
而真巴赫只有 0.011/100 对 —— 音高对了，声部进行却是初学者水平。
这跟 v9 当年"神经模型 + 事后规则修补"的困境一样，正确做法是**在解码时引导**
（项目在 v9 上已确立的做法：多候选 → 乐理评分 → 择优）。

本实验
----
不看真值，只用乐理打分选候选：
    1. 和弦音吻合（用给定的和声条件：func/type/root → 和弦音）
    2. 声部进行违规惩罚（平行五/八度、上三声部间距超八度、声部交越、超音域）
然后**再用真值算准确率**，检查"乐理打分"是否既降低违规、又不损失音高准确率。

用法: python analyze_v10_decode.py [--candidates 8] [--temp 1.0] [--n 48]
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn.functional as F

from constants import CHORD_INTERVALS
from model.satb_diffusion import SATBDiffusion, SatbVocab, voiceleading_metrics
from train_v10_satb import (COND_KEYS, N_VOICES, VOICE_NAMES, group_split,
                            load_corpus, DATA)

ID2TYPE = {0: 'M', 1: 'm', 2: 'dim', 3: 'dom7', 4: 'aug', 5: 'm7', 6: 'dim7', 7: 'REST'}


def chord_tones(cond, T, dev):
    """每切片的和弦音 pc 集合（和声未知则返回 None）。"""
    out = []
    func, typ, root = cond['func'][0], cond['type'][0], cond['root'][0]
    for t in range(T):
        if int(root[t]) >= 12:
            out.append(None)
            continue
        ivs = CHORD_INTERVALS.get(ID2TYPE.get(int(typ[t]), 'M'), [0, 4, 7])
        out.append({(int(root[t]) + iv) % 12 for iv in ivs})
    return out


def score_candidate(pitch, vocab, cts):
    """乐理打分（越高越好）：和弦音吻合率 − 声部进行违规惩罚。无真值参与。"""
    V, T = pitch.shape
    hit = tot = 0
    for v in range(V):
        for t in range(T):
            p = int(pitch[v, t])
            if p >= vocab.n_pitch or cts[t] is None:
                continue
            tot += 1
            if (p + vocab.lo) % 12 in cts[t]:
                hit += 1
    chord = hit / max(tot, 1)
    m = voiceleading_metrics(torch.where(
        pitch < vocab.n_pitch, pitch + vocab.lo, torch.full_like(pitch, -1))[None])
    pairs = max(m['pairs'], 1)
    pen = (2.0 * m['parallel_5'] + 2.0 * m['parallel_8'] + 0.5 * m['spacing']
           + 0.5 * m['crossing'] + 0.3 * m['range_violation']) / pairs
    return chord - pen, chord, pen


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='v10_chorales.pt')
    ap.add_argument('--corpus', type=str, default='chorales')
    ap.add_argument('--d', type=int, default=480)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8, help='注意力头数（须与训练时一致）')
    ap.add_argument('--win', type=int, default=64)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--n', type=int, default=48, help='评估窗口数')
    ap.add_argument('--candidates', type=int, default=8)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='data/processed/v10_decode_experiment.json',
                    help='产物路径（每次实验写自己的文件，别覆盖历史口径）')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = SatbVocab()
    windows = load_corpus(args.corpus, vocab, args.win, args.stride)
    _, vl, _ = group_split(windows, 0.08)
    rng = random.Random(args.seed)
    sample = rng.sample(vl, min(args.n, len(vl)))
    model = SATBDiffusion(d=args.d, h=args.heads, layers=args.layers, vocab=vocab).to(dev)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=dev, weights_only=True))
    model.eval()

    stats = {tag: Counter() for tag in ('argmax', 'best_of_k', 'random_cand')}
    acc = {tag: [0, 0] for tag in stats}
    for w in sample:
        pitch = w['pitch'][None].to(dev)
        rhythm = w['rhythm'][None].to(dev)
        known = torch.zeros_like(pitch, dtype=torch.bool)
        known[:, 0] = True                                  # 只给 Soprano
        cond = {k: w[k][None].to(dev) for k in COND_KEYS}
        T = pitch.shape[2]
        cts = chord_tones(cond, T, dev)
        real = (pitch[0] < vocab.n_pitch)

        # 候选 0: 单步 argmax（基线）
        gp0, _ = model.generate(pitch, rhythm, known, cond, steps=1, argmax=True)
        # 候选 1..K-1: 单步采样（温度控制多样性）
        cands = [gp0[0]]
        for _ in range(args.candidates - 1):
            gp, _ = model.generate(pitch, rhythm, known, cond, steps=1, argmax=False,
                                   temp=args.temp, seed=rng.randint(0, 10 ** 6))
            cands.append(gp[0])

        scored = [score_candidate(c, vocab, cts) for c in cands]
        pick_best = max(range(len(cands)), key=lambda i: scored[i][0])
        pick_rand = rng.randrange(len(cands))

        for tag, cand in (('argmax', cands[0]), ('best_of_k', cands[pick_best]),
                          ('random_cand', cands[pick_rand])):
            hit = tot = 0
            for v in (1, 2, 3):
                hit += ((cand[v] == pitch[0, v]) & real[v]).sum().item()
                tot += int(real[v].sum().item())
            acc[tag][0] += hit
            acc[tag][1] += tot
            midi = torch.where(cand < vocab.n_pitch, cand + vocab.lo,
                               torch.full_like(cand, -1))
            m = voiceleading_metrics(midi[None])
            for k, v_ in m.items():
                stats[tag][k] += v_

    print(f'样本 {len(sample)} 窗 × {args.candidates} 候选 (temp {args.temp})')
    print(f'{"策略":12s} {"音高准确率":>10s} {"平行五/100":>11s} {"平行八度/100":>12s} '
          f'{"间距>8度/100":>12s} {"交越/100":>10s}')
    for tag in ('argmax', 'best_of_k', 'random_cand'):
        s = stats[tag]
        p = max(s['pairs'], 1)
        a = acc[tag][0] / max(acc[tag][1], 1)
        print(f'{tag:12s} {a:10.3f} {100*s["parallel_5"]/p:11.3f} '
              f'{100*s["parallel_8"]/p:12.3f} {100*s["spacing"]/p:12.3f} '
              f'{100*s["crossing"]/p:10.3f}')
    json.dump({'config': vars(args),
               **{tag: {'pitch_acc': acc[tag][0] / max(acc[tag][1], 1), **dict(stats[tag])}
                  for tag in stats}},
              open(ROOT / args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
