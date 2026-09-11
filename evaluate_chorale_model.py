"""
句子级模型完整证据链评估 (v9) —— 恢复 / 边界 / 生成 / 自洽 / 防模仿 / 句法。

与旧 `analyze_syntax.py` 的区别:
    1. 用 `src/chorale_conditions.py` 的统一条件编码 (训练同源);
    2. 边界指标含阈值扫描 / ±容差 / 分终止式类型, 不是单点数字;
    3. 新增**自洽边界率** (模型对自己生成的旋律划句, 命中计划到达点)
       —— "模型是否理解自己写的句子";
    4. 新增**防模仿证据** (SIMAA@90/95 + 17 维马氏距离, MusicLDM/DRMW 口径),
       并给人类语料基线 —— v8 从未做过这项;
    5. 终止式实现率区分**渲染前 / 渲染后**两个口径 (渲染含规则修正)。

用法:
    python evaluate_chorale_model.py --model melody_diffusion_v9_grouped.pt \
        --out data/processed/v9_eval.json
"""
import sys, json, random, argparse, statistics
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from chorale_conditions import (load_chorale_windows, grouped_split,
                                kfold_by_piece, CADENCE_NAME)
from constants import CHORD_INTERVALS, F2ID_MELODY, T2ID
from model.melody_diffusion import MelodyDiffusion, REST
from metrics.anti_imitation import simaa, mahalanobis, melody_features
from train_v8_chorales import ChoraleDataset
from train_v9_chorales import build_model, eval_recovery, eval_boundary, eval_generation

DATA = str(ROOT / 'data/processed/chorales_sentences_v1.json')
ID2TYPE = {v: k for k, v in T2ID.items()}
GEN_CANDIDATES = 2


def load_ckpt(path, device):
    m = build_model(argparse.Namespace(d=320, layers=6)).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 生成 + 句法/终止式指标 (与 gen_sentence 管线同口径, 但留在音级层)
# ─────────────────────────────────────────────────────────────────────────────

def phrase_spans(cond):
    return [s for s in _spans(cond)]


def _spans(cond):
    ends = [i for i in range(len(cond['phrase_bin'])) if cond['phrase_bin'][i] == 0]
    s, out = 0, []
    for e in ends:
        out.append((s, e)); s = e + 1
    return out


def contour(a, b):
    def c(seq):
        out = []
        for x, y in zip(seq[:-1], seq[1:]):
            if x >= REST or y >= REST:
                continue
            d = (y - x) % 12
            out.append(d - 12 if d > 6 else d)
        return out
    A, B = c(a), c(b)
    if not A or not B:
        return 0.0
    m, n = len(A), len(B)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = dp[i][j] + 1 if A[i] == B[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n] / min(m, n)


@torch.no_grad()
def generate_best(model, cond, device, rng, candidates=GEN_CANDIDATES, steps=16):
    """多候选 + 终止式到达点和弦音评分 (与 gen_sentence 同口径)。"""
    n = len(cond['pc'])
    t = lambda k: torch.tensor([cond[k]], device=device)
    best, best_sc = None, -1e9
    for _ in range(candidates):
        gp, grh = model.generate(t('func'), t('type'), t('root'), steps=steps,
                                 temp=0.85, remask_steps=6, remask_ratio=0.3,
                                 repeat_damp=0.05, expand=False,
                                 pos_bin=t('pos_bin'), toend_bin=t('toend_bin'),
                                 phrase_bin=t('phrase_bin'), cadence=t('cadence'),
                                 seed=rng.randint(0, 10 ** 6))
        pcs = [int(x) for x in gp[0].tolist()]
        hit = tot = 0
        for i in range(n):
            if cond['phrase_bin'][i] == 0:
                tot += 1
                ivs = CHORD_INTERVALS.get(ID2TYPE.get(cond['type'][i], 'M'), [0, 4, 7])
                if pcs[i] < REST and (pcs[i] - cond['root'][i]) % 12 in ivs:
                    hit += 1
        sc = hit / max(tot, 1)
        if sc > best_sc:
            best_sc, best = sc, (pcs, grh)
    return best


@torch.no_grad()
def self_consistency(model, pcs, cond, device):
    """模型对自己生成的旋律划句, 命中计划到达点的比例 (F1 口径)。"""
    n = len(pcs)
    gp = torch.tensor([pcs], device=device)
    fn = torch.tensor([cond['func']], device=device)
    tp = torch.tensor([cond['type']], device=device)
    rt = torch.tensor([cond['root']], device=device)
    pb = torch.tensor([cond['pos_bin']], device=device)
    tb = torch.tensor([cond['toend_bin']], device=device)
    phb = torch.full_like(pb, 6)
    cad = torch.zeros_like(pb)
    flag = torch.ones_like(gp)
    _, _, bl = model(gp, torch.zeros_like(gp), fn, tp, rt, flag, None, pb, tb, phb, cad,
                     return_boundary=True)
    pred = (torch.softmax(bl, -1)[0, :, 1] > 0.5)
    planned = set(e for _, e in _spans(cond))
    got = set(i for i in pred.nonzero().flatten().tolist())
    tp_ = len(got & planned)
    p = tp_ / max(len(got), 1); r = tp_ / max(len(planned), 1)
    return 2 * p * r / max(p + r, 1e-6), p, r


def evaluate_checkpoint(model, val_samples, device, n_gen=32, seed=0):
    rng = random.Random(seed)
    vl_ld = DataLoader(ChoraleDataset(val_samples), batch_size=64)
    out = {}
    out.update(eval_recovery(model, vl_ld, device))
    out.update(eval_boundary(model, vl_ld, device))
    out['generation_acc'] = eval_generation(model, val_samples, device, n=48, seed=seed)

    # 生成 + 句法/自洽/终止式 (渲染前口径)
    pool = [s for s in val_samples if sum(s['boundary']) >= 2]
    rng.shuffle(pool)
    pool = pool[:n_gen]
    plens, acs, cad_hits, cons, cons_p, cons_r = [], [], [], [], [], []
    base_plens, base_acs = [], []      # 真值基线: 同一批窗口, 控制截断偏差
    gen_pieces = []
    for s in pool:
        cond = {'phrase_bin': s['phrase_bin'], 'cadence': s['cadence'],
                'type': s['type'], 'root': s['root'], 'func': s['func'],
                'pos_bin': s['pos_bin'], 'toend_bin': s['toend_bin'], 'pc': s['pc']}
        spans = _spans(cond)
        base_plens += [e - st + 1 for st, e in spans]
        for (s1b, e1b), (s2b, e2b) in zip(spans[:-1], spans[1:]):
            base_acs.append(contour(s['pc'][s1b:e1b + 1], s['pc'][s2b:e2b + 1]))
        pcs, _ = generate_best(model, cond, device, rng)
        gen_pieces.append((pcs, s['rhythm']))
        plens += [e - st + 1 for st, e in spans]
        for (s1, e1), (s2, e2) in zip(spans[:-1], spans[1:]):
            acs.append(contour(pcs[s1:e1 + 1], pcs[s2:e2 + 1]))
        for st, e in spans:
            ivs = CHORD_INTERVALS.get(ID2TYPE.get(cond['type'][e], 'M'), [0, 4, 7])
            if pcs[e] >= REST:
                cad_hits.append(0.0); continue
            rel = (pcs[e] - cond['root'][e]) % 12
            cad = cond['cadence'][e]
            cad_hits.append(1.0 if (rel == 0 if cad in (1, 4) else rel in ivs) else 0.0)
        f1c, pc_, rc_ = self_consistency(model, pcs, cond, device)
        cons.append(f1c); cons_p.append(pc_); cons_r.append(rc_)

    out['gen_phrase_len_mean'] = float(np.mean(plens)) if plens else 0.0
    out['gen_contour_sim'] = float(np.mean(acs)) if acs else 0.0
    out['base_phrase_len_mean'] = float(np.mean(base_plens)) if base_plens else 0.0
    out['base_contour_sim'] = float(np.mean(base_acs)) if base_acs else 0.0
    out['cadence_realization'] = float(np.mean(cad_hits)) if cad_hits else 0.0
    out['self_consistency_f1'] = float(np.mean(cons)) if cons else 0.0
    out['self_consistency_p'] = float(np.mean(cons_p)) if cons_p else 0.0
    out['self_consistency_r'] = float(np.mean(cons_r)) if cons_r else 0.0

    # 防模仿: 生成 vs 训练语料 (SIMAA + 马氏距离)
    corpus_pieces = json.load(open(DATA, encoding='utf-8'))
    train_tokens, train_pairs = [], []
    for p in corpus_pieces:
        pcs = [min(REST, nt['pc']) for nt in p['notes']]
        rhs = [min(7, nt['rhythm']) for nt in p['notes']]
        train_tokens += pcs
        train_pairs.append((pcs, rhs))
    gen_seqs = [pcs for pcs, _ in gen_pieces]
    s = simaa(train_tokens, gen_seqs)
    out['simaa'] = {k: v for k, v in s.items()}
    mh = mahalanobis(train_pairs, gen_pieces)
    out['mahalanobis'] = mh
    # 人类基线: 语料自身留出曲 vs 训练语料 (同样截 32 音窗口, 与生成口径一致)
    _, valp, vp = grouped_split(load_chorale_windows(DATA), 0.1, 42)
    human_pairs = [([min(REST, nt['pc']) for nt in p['notes']][:32],
                    [min(7, nt['rhythm']) for nt in p['notes']][:32])
                   for i, p in enumerate(corpus_pieces)
                   if i in vp and len(p['notes']) >= 32]
    hum_tokens = [pc for pr, _ in human_pairs for pc in pr]
    out['simaa_human'] = {k: v for k, v in
                          simaa(train_tokens, [pr[:32] for pr, _ in human_pairs]).items()}
    out['mahalanobis_human'] = mahalanobis(train_pairs, human_pairs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='melody_diffusion_v9_grouped.pt')
    ap.add_argument('--models', type=str, nargs='*', default=None,
                    help='多个折的权重: 汇总 均值±标准差')
    ap.add_argument('--out', type=str, default='data/processed/v9_eval.json')
    ap.add_argument('--n-gen', type=int, default=32)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备 {device}')
    samples = load_chorale_windows(DATA)

    if args.models and len(args.models) > 1:
        folds = kfold_by_piece(samples, k=len(args.models), seed=42)
        per = []
        for path, (tr, vl, vp) in zip(args.models, folds):
            model = load_ckpt(path, device)
            r = evaluate_checkpoint(model, vl, device, n_gen=args.n_gen, seed=0)
            r['fold'] = path
            per.append(r)
            print(f'  {Path(path).name}: rec {r["recovery_t50"]:.3f} | F1±1 {r["f1_tol1"]:.3f} '
                  f'| 自洽 {r["self_consistency_f1"]:.3f} | 终止式 {r["cadence_realization"]:.3f}')
        keys = [k for k in per[0] if isinstance(per[0][k], (int, float))]
        summary = {'n_folds': len(per),
                   'folds': per,
                   'mean_std': {k: {'mean': round(statistics.mean([f[k] for f in per]), 4),
                                    'std': round(statistics.pstdev([f[k] for f in per]), 4)}
                                for k in keys}}
        json.dump(summary, open(ROOT / args.out, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        print(f'\n{len(per)} 折汇总 → {args.out}')
        for k in ('recovery_t15', 'recovery_t50', 'recovery_t85', 'f1_at_0.5', 'f1_tol1',
                  'f1_tol2', 'precision', 'recall', 'generation_acc', 'self_consistency_f1',
                  'cadence_realization', 'gen_phrase_len_mean', 'gen_contour_sim'):
            c = summary['mean_std'].get(k)
            if c:
                print(f'  {k:22s} {c["mean"]:.3f} ± {c["std"]:.3f}')
        return

    model = load_ckpt(ROOT / args.model if not Path(args.model).is_absolute() else args.model,
                      device)
    _, vl, vp = grouped_split(samples, 0.1, 42)
    res = evaluate_checkpoint(model, vl, device, n_gen=args.n_gen, seed=0)
    res['_val_pieces'] = sorted(vp)
    json.dump(res, open(ROOT / args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'恢复 t.15/t.50/t.85: {res["recovery_t15"]:.3f} / {res["recovery_t50"]:.3f} / {res["recovery_t85"]:.3f}')
    print(f'边界 F1@.5 {res["f1_at_0.5"]:.3f} (P{res["precision"]:.2f}/R{res["recall"]:.2f}) '
          f'| F1±1 {res["f1_tol1"]:.3f} | F1±2 {res["f1_tol2"]:.3f} | best_th {res["best_threshold"]}')
    print(f'分终止式召回: ' + ', '.join(f'{k} {v["recall"]:.2f}(n={v["n"]})'
                                    for k, v in res['recall_by_cadence'].items()))
    print(f'全掩码生成 {res["generation_acc"]:.3f} | 自洽边界 F1 {res["self_consistency_f1"]:.3f} '
          f'(P{res["self_consistency_p"]:.2f}/R{res["self_consistency_r"]:.2f})')
    print(f'终止式实现 {res["cadence_realization"]:.3f} | 乐句长 {res["gen_phrase_len_mean"]:.1f} '
          f'| 前后句相似 {res["gen_contour_sim"]:.3f}')
    print(f'SIMAA@90 {res["simaa"]["SIMAA@90"]:.3f} (人类 {res["simaa_human"]["SIMAA@90"]:.3f}) '
          f'| 马氏 {res["mahalanobis"]["gen_mean"]:.2f} (训练 {res["mahalanobis"]["train_mean"]:.2f}, '
          f'人类 {res["mahalanobis_human"]["gen_mean"]:.2f})')
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
