"""
句法评估指标 (步骤 5) —— 量化"模型是否理解句子"。

指标:
    1. 边界 F1      : 给定真实旋律 (遮蔽结构流), 模型预测乐句末的 P/R/F1;
    2. 终止式实现率  : 生成旋律在计划到达点上的和弦音吻合率
                       (全终止要求落主音 = 和弦根音);
    3. 自洽边界率    : 模型对**自己生成的**旋律做边界判断,
                       命中计划到达点的比例 (生成是否"听起来有句法");
    4. 句子长度分布  : 生成乐句长度 vs 众赞歌语料;
    5. 前后句关系    : 相邻乐句轮廓相似度 (乐段性) vs 语料基线。

用法: python analyze_syntax.py [--n 24]
"""
import sys, os, json, random, argparse, statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from constants import CHORD_INTERVALS, F2ID_MELODY, T2ID
from model.melody_diffusion import MelodyDiffusion, REST
from train_v8_chorales import load_chorale_windows, ChoraleDataset, CADENCE_ID
from gen_sentence import piece_conditions, load_model, ID2TYPE

ID2TYPE_LOCAL = ID2TYPE


def boundary_f1(model, loader, device, threshold=0.5):
    tp = fp = fn_ = 0
    with torch.no_grad():
        for b in loader:
            fn, tp_, rt, pc, rh, pb, tb, phb, cad, bd = [x.to(device) for x in b]
            flag = torch.ones_like(pc)
            phb = torch.full_like(phb, 6)
            cad = torch.zeros_like(cad)
            _, _, bl = model(pc, rh, fn, tp_, rt, flag, None, pb, tb, phb, cad,
                             return_boundary=True)
            pred = (torch.softmax(bl, -1)[..., 1] > threshold).long()
            tp += ((pred == 1) & (bd == 1)).sum().item()
            fp += ((pred == 1) & (bd == 0)).sum().item()
            fn_ += ((pred == 0) & (bd == 1)).sum().item()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn_, 1)
    return 2 * prec * rec / max(prec + rec, 1e-6), prec, rec


def contour_lcs(a, b):
    def cont(seq):
        out = []
        for x, y in zip(seq[:-1], seq[1:]):
            if x >= REST or y >= REST:
                continue
            d = (y - x) % 12
            out.append(d - 12 if d > 6 else d)
        return out
    A, B = cont(a), cont(b)
    if not A or not B:
        return 0.0
    m, n = len(A), len(B)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if A[i] == B[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n] / min(m, n)


def phrase_spans(cond, n):
    ends = [i for i in range(n) if cond['phrase_bin'][i] == 0]
    spans, s = [], 0
    for e in ends:
        spans.append((s, e)); s = e + 1
    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=24)
    parser.add_argument('--out', type=str, default='data/processed/syntax_report.json')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(42); torch.manual_seed(42)      # 固定数据划分/生成种子

    model = load_model(device)
    data = json.load(open(ROOT / 'data/processed/chorales_sentences_v1.json', encoding='utf-8'))

    # ── 1. 边界 F1 (验证集) ──
    samples = load_chorale_windows(str(ROOT / 'data/processed/chorales_sentences_v1.json'))
    sp = int(len(samples) * 0.9)
    vl = torch.utils.data.DataLoader(ChoraleDataset(samples[sp:]), batch_size=64)
    f1, prec, rec = boundary_f1(model, vl, device)
    print(f'1. 边界 F1 (真实旋律, 结构遮蔽): {f1:.3f} (P {prec:.3f} / R {rec:.3f})')

    # ── 语料基线: 句子长度 / 前后句关系 ──
    corpus_plens, corpus_acs = [], []
    for p in data:
        cond = piece_conditions(p)
        n = len(p['notes'])
        spans = phrase_spans(cond, n)
        pcs = [nt['pc'] for nt in p['notes']]
        for s, e in spans:
            corpus_plens.append(e - s + 1)
        for (s1, e1), (s2, e2) in zip(spans[:-1], spans[1:]):
            corpus_acs.append(contour_lcs(pcs[s1:e1 + 1], pcs[s2:e2 + 1]))

    # ── 2-5. 生成 + 指标 ──
    pool = [p for p in data if len(p['notes']) >= 32]
    rng = random.Random(7)
    gen_plens, gen_acs, cad_hits = [], [], []
    arrival_consistency = []
    for k in range(args.n):
        p = rng.choice(pool)
        cond = piece_conditions(p)
        n = len(p['notes'])
        fn = torch.tensor([cond['func']], device=device)
        tp = torch.tensor([cond['type']], device=device)
        rt = torch.tensor([cond['root']], device=device)
        pb = torch.tensor([cond['pos_bin']], device=device)
        tb = torch.tensor([cond['toend_bin']], device=device)
        phb = torch.tensor([cond['phrase_bin']], device=device)
        cad = torch.tensor([cond['cadence']], device=device)
        # 多候选 + 终止式感知评分 (与 gen_sentence 管线同口径)
        best, best_sc = None, -1e9
        for _c in range(2):
            with torch.no_grad():
                gp, _ = model.generate(fn, tp, rt, steps=16, temp=0.85, remask_steps=6,
                                       remask_ratio=0.3, repeat_damp=0.05, expand=False,
                                       pos_bin=pb, toend_bin=tb, phrase_bin=phb,
                                       cadence=cad, seed=rng.randint(0, 10**6))
            _pcs = [int(x) for x in gp[0].tolist()]
            hit = tot_a = 0
            for i in range(n):
                if cond['phrase_bin'][i] == 0:
                    tot_a += 1
                    ivs = CHORD_INTERVALS.get(ID2TYPE_LOCAL.get(cond['type'][i], 'M'), [0, 4, 7])
                    if _pcs[i] < REST and (_pcs[i] - cond['root'][i]) % 12 in ivs:
                        hit += 1
            _sc = hit / max(tot_a, 1)
            if _sc > best_sc:
                best_sc, best = _sc, _pcs
        pcs = best
        spans = phrase_spans(cond, n)
        gen_plens += [e - s + 1 for s, e in spans]
        for (s1, e1), (s2, e2) in zip(spans[:-1], spans[1:]):
            gen_acs.append(contour_lcs(pcs[s1:e1 + 1], pcs[s2:e2 + 1]))
        # 终止式实现率
        for s, e in spans:
            cad_t = cond['cadence'][e]
            if pcs[e] >= REST:
                cad_hits.append(0.0); continue
            ivs = CHORD_INTERVALS.get(ID2TYPE_LOCAL.get(cond['type'][e], 'M'), [0, 4, 7])
            rel = (pcs[e] - cond['root'][e]) % 12
            if cad_t in (1, 4):                     # 全/变格: 要求主音 (根音)
                cad_hits.append(1.0 if rel == 0 else 0.0)
            else:                                   # 半/阻碍: 和弦音即可
                cad_hits.append(1.0 if rel in ivs else 0.0)
        # 自洽边界: 模型对自己生成旋律的边界判断命中计划到达点
        gflag = torch.ones(1, n, dtype=torch.long, device=device)
        with torch.no_grad():
            _, _, bl = model(gp, torch.zeros_like(gp), fn, tp, rt, gflag, None,
                             pb, tb, torch.full_like(phb, 6), torch.zeros_like(cad),
                             return_boundary=True)
        pred = (torch.softmax(bl, -1)[0, :, 1] > 0.5)
        planned = set(e for _, e in spans)
        hit_planned = sum(1 for i in pred.nonzero().flatten().tolist() if i in planned)
        arrival_consistency.append(hit_planned / max(len(planned), 1))

    report = {
        'boundary_f1': f1, 'boundary_precision': prec, 'boundary_recall': rec,
        'cadence_realization': float(np.mean(cad_hits)),
        'self_consistency': float(np.mean(arrival_consistency)),
        'gen_phrase_len_mean': float(np.mean(gen_plens)),
        'corpus_phrase_len_mean': float(np.mean(corpus_plens)),
        'gen_ac_sim': float(np.mean(gen_acs)) if gen_acs else 0.0,
        'corpus_ac_sim': float(np.mean(corpus_acs)) if corpus_acs else 0.0,
    }
    print(f"2. 终止式实现率: {report['cadence_realization']:.3f}")
    print(f"3. 自洽边界率:   {report['self_consistency']:.3f}")
    print(f"4. 生成乐句长度: {report['gen_phrase_len_mean']:.1f} 音 (语料 {report['corpus_phrase_len_mean']:.1f})")
    print(f"5. 前后句轮廓相似度: 生成 {report['gen_ac_sim']:.3f} vs 语料 {report['corpus_ac_sim']:.3f}")
    json.dump(report, open(ROOT / args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
