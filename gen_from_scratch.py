"""
从零生成（v11）—— 不给旋律、不给和声标签，模型自己写出四个声部。

与 `gen_satb.py`（配和声）的区别
------------------------------------------------
    gen_satb : 给 Soprano + 和声标签 → 补 Alto/Tenor/Bass     （实现 / realization）
    本脚本   : 只给"规则生成的乐句计划" → 生成全部声部 + 和声   （生成 / generation）

外部只提供**曲式模板**（4 个乐句、每句约 11 切片、前半句半终止、末句全终止）——
这是众赞歌的通用骨架，不来自任何一首巴赫作品；调性、和声、四个声部全部由模型生成。

评估：自由生成**没有真值**，"逐音准确率"失去意义。改用音乐性指标：
    1. 调性一致性      —— Krumhansl-Schmuckler 置信度（越高越像"在一个调里"）
    2. 终止式落点      —— 乐句末的**低音**是否落在该终止式应有的音级上
    3. 纵向音响成和弦率 —— 每切片的音级集合能否被某个已知和弦（三/七和弦）解释
    4. 声部进行违规     —— 平行五/八度、交越、间距、越界（配真巴赫基线）
    5. 防模仿          —— SIMAA@90（生成旋律 vs 训练语料的记忆检测）
全部与**真众赞歌**的同一批指标对比。

用法
------------------------------------------------
    python gen_from_scratch.py --ckpt v11_scratch.pt --n 5          # 生成并评估
    python gen_from_scratch.py --ckpt v11_scratch.pt --n 1 --bars 24
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from constants import CHORD_INTERVALS
from metrics.anti_imitation import simaa
from metrics.theory_metrics import key_confidence
from model.satb_diffusion import N_VOICES, SATBDiffusion, SatbVocab, voiceleading_metrics
from gen_satb import write_midi
from train_v10_satb import COND_KEYS, DATA, HOLD, RHYTHM_VALUES, VOICE_NAMES
from train_harmony_planner import (N_FUNC_LABEL, N_TYPE_LABEL, HarmonyPlanner,
                                   sample_plan_token)

VOCAB = SatbVocab()
CHORD_TEMPLATES = []
for root in range(12):
    for ivs in CHORD_INTERVALS.values():
        CHORD_TEMPLATES.append({(root + iv) % 12 for iv in ivs})


@torch.no_grad()
def plan_harmony(planner, cond, T, device, seed=0):
    """用规划器自回归生成和弦计划，写回 cond 的 func/type/root。"""
    torch.manual_seed(seed)
    pf = torch.full((1, 1), -1, dtype=torch.long, device=device)
    pt = torch.full((1, 1), -1, dtype=torch.long, device=device)
    pr = torch.full((1, 1), -1, dtype=torch.long, device=device)
    fs, ts, rs = [], [], []
    for i in range(T):
        c = {k: cond[k][:, i:i + 1] for k in cond}
        lf, lt, lr = planner(c, pf, pt, pr)
        fs.append(sample_plan_token(lf[0, -1], N_FUNC_LABEL))
        ts.append(sample_plan_token(lt[0, -1], N_TYPE_LABEL))
        rs.append(sample_plan_token(lr[0, -1], 12))                # root 0-11
        pf = torch.cat([pf, torch.tensor([[fs[-1]]], device=device)], 1)
        pt = torch.cat([pt, torch.tensor([[ts[-1]]], device=device)], 1)
        pr = torch.cat([pr, torch.tensor([[rs[-1]]], device=device)], 1)
    out = dict(cond)
    out['func'] = torch.tensor([fs], device=device)
    out['type'] = torch.tensor([ts], device=device)
    out['root'] = torch.tensor([rs], device=device)
    return out


def build_plan(n_slices: int, vocab: SatbVocab, device='cpu', n_phrases: int = 4):
    """规则曲式模板：n_phrases 个乐句，前半句半终止、末句全终止。"""
    ends = [int(round((i + 1) * n_slices / n_phrases)) - 1 for i in range(n_phrases)]
    ends[-1] = n_slices - 1
    phrase_bin, cadence, pos_bin, toend_bin = [], [], [], []
    for i in range(n_slices):
        nxt = next(k for k, e in enumerate(ends) if e >= i)
        phrase_bin.append(min(max(0, ends[nxt] - i), 5))
        cadence.append(2 if nxt < n_phrases - 1 and nxt == (n_phrases - 1) // 2 else
                       (1 if nxt == n_phrases - 1 else 0))
        pos_bin.append(min(int(i / n_slices * 8), 7))
        rem = 1.0 - (i + 1) / n_slices
        toend_bin.append(0 if rem < 0.06 else (1 if rem < 0.15 else (2 if rem < 0.4 else 3)))
    T = n_slices
    return {
        'func': torch.full((1, T), 7, dtype=torch.long, device=device),      # 未知
        'type': torch.full((1, T), 9, dtype=torch.long, device=device),
        'root': torch.full((1, T), 12, dtype=torch.long, device=device),
        'pos_bin': torch.tensor([pos_bin], device=device),
        'toend_bin': torch.tensor([toend_bin], device=device),
        'phrase_bin': torch.tensor([phrase_bin], device=device),
        'cadence': torch.tensor([cadence], device=device),
        'style': torch.zeros(1, T, dtype=torch.long, device=device),
    }, ends


# ─────────────────────────────────────────────────────────────────────────────
# 音乐性评估
# ─────────────────────────────────────────────────────────────────────────────

def chord_explainable(pcs: list[int]) -> bool:
    """该切片的音级集合能否被某个已知和弦解释（允许重复音与省略）。"""
    s = set(pcs)
    return any(s <= t for t in CHORD_TEMPLATES)


def _key_stats(pcs_per_piece: list[list[int]]) -> dict:
    """逐曲调性判定 → 四个口径（避免只报一个有符号均值）。

    只看**主音**是否为 C（`k['key'].split()[0] == 'C'`）—— 旧写法用
    `startswith('C')` 会把 `C# major/minor`（差半音）也算成"在 C 调"。
    `key_conf_chorale` 保留为有符号 C 度（与旧报告可比），
    但它的绝对值会被"几首跑到了 G/属调"主导，读的时候必须配
    `key_c_ratio`（落在 C 的比例）与 `key_conf_abs_mean`（调性有多明确）一起看。
    """
    confs, keys = [], []
    for pcs in pcs_per_piece:
        if not pcs:
            continue
        hist = np.zeros(12)
        for pc in pcs:
            hist[pc] += 1
        k = key_confidence(hist)
        confs.append(k['confidence'])
        keys.append(k['key'])
    if not confs:
        return {'key_conf_chorale': 0.0, 'key_c_ratio': 0.0,
                'key_conf_abs_mean': 0.0, 'key_dist': {}}
    is_c = [k.split()[0] == 'C' for k in keys]
    signed = [cf if c else -cf for cf, c in zip(confs, is_c)]
    return {
        'key_conf_chorale': float(np.mean(signed)),
        'key_c_ratio': float(np.mean([float(c) for c in is_c])),
        'key_conf_abs_mean': float(np.mean(confs)),
        'key_dist': dict(Counter(keys).most_common()),
    }


def evaluate_generated(gen_list, ends_list, ref_metrics: dict) -> dict:
    """gen_list: [(pitch[4,T] 词表索引, rhythm[4,T])]；ends_list: 各曲的乐句末切片。"""
    pcs_per_piece, chord_ok, chord_tot = [], 0, 0
    vl_tot = Counter()
    cad_hit = cad_tot = 0
    sop_lines = []
    for (pitch, rhythm), ends in zip(gen_list, ends_list):
        V, T = pitch.shape
        pcs_all = []
        for t in range(T):
            ps = [int(pitch[v, t]) for v in range(V) if int(pitch[v, t]) < VOCAB.n_pitch]
            if not ps:
                continue
            pcs = [(p + VOCAB.lo) % 12 for p in ps]
            pcs_all += pcs
            chord_tot += 1
            chord_ok += int(chord_explainable(pcs))
        pcs_per_piece.append(pcs_all)
        # 终止式落点: 乐句末切片的最低音是否在 主/属 音级上
        for e in ends:
            ps = [(int(pitch[v, e]), v) for v in range(V) if int(pitch[v, e]) < VOCAB.n_pitch]
            if not ps:
                continue
            bass = min(ps)[0]
            bass_pc = (bass + VOCAB.lo) % 12
            cad_tot += 1
            cad_hit += int(bass_pc in (0, 7))          # 主音或属音（C 大调）
        midi = torch.where(pitch < VOCAB.n_pitch, pitch + VOCAB.lo,
                           torch.full_like(pitch, -1))
        for k, v in voiceleading_metrics(midi[None]).items():
            vl_tot[k] += v
        sop = [int(pitch[0, t]) for t in range(T)]
        sop_lines.append([(p + VOCAB.lo) % 12 if p < VOCAB.n_pitch else 12 for p in sop])
    pairs = max(vl_tot['pairs'], 1)
    return {
        **_key_stats(pcs_per_piece),
        'chord_explainable': chord_ok / max(chord_tot, 1),
        'cadence_bass_on_tonic_or_dom': cad_hit / max(cad_tot, 1),
        'parallel_5_per100': 100 * vl_tot['parallel_5'] / pairs,
        'parallel_8_per100': 100 * vl_tot['parallel_8'] / pairs,
        'crossing_per100': 100 * vl_tot['crossing'] / pairs,
        'spacing_per100': 100 * vl_tot['spacing'] / pairs,
        'ref': ref_metrics,
    }


def corpus_reference() -> dict:
    """真众赞歌的同一批指标（基线）。"""
    pieces = json.load(open(DATA / 'chorales_satb_v2.json', encoding='utf-8'))
    pcs_per_piece, chord_ok, chord_tot = [], 0, 0
    cad_hit = cad_tot = 0
    vl_tot = Counter()
    sop_lines = []
    for p in pieces:
        slices = p['slices']
        T = len(slices)
        pcs_all = []
        for s in slices:
            ps = [(m, v) for v, m in s['midi'].items() if m is not None]
            if not ps:
                continue
            pcs = [m % 12 for m, _ in ps]
            pcs_all += pcs
            chord_tot += 1
            chord_ok += int(chord_explainable(pcs))
        pcs_per_piece.append(pcs_all)
        # 终止式落点（与 evaluate_generated 同一条规则：乐句末最低音在主/属音级）
        for ph in p['phrases']:
            j = ph['end_slice']
            if not (0 <= j < T):
                continue
            ps = [m for m in slices[j]['midi'].values() if m is not None]
            if not ps:
                continue
            cad_tot += 1
            cad_hit += int(min(ps) % 12 in (0, 7))
        midi = torch.full((1, N_VOICES, T), -1, dtype=torch.long)
        for t, s in enumerate(slices):
            for v, name in enumerate(VOICE_NAMES):
                m = s['midi'].get(name)
                if m is not None:
                    midi[0, v, t] = m
        for k, v in voiceleading_metrics(midi).items():
            vl_tot[k] += v
        sop_lines.append([s['midi']['S'] % 12 if s['midi']['S'] is not None else 12
                          for s in slices])
    pairs = max(vl_tot['pairs'], 1)
    flat = [pc for line in sop_lines for pc in line]
    sim = simaa(flat, sop_lines)
    return {
        **_key_stats(pcs_per_piece),
        'chord_explainable': chord_ok / max(chord_tot, 1),
        'cadence_bass_on_tonic_or_dom': cad_hit / max(cad_tot, 1),
        'parallel_5_per100': 100 * vl_tot['parallel_5'] / pairs,
        'parallel_8_per100': 100 * vl_tot['parallel_8'] / pairs,
        'crossing_per100': 100 * vl_tot['crossing'] / pairs,
        'spacing_per100': 100 * vl_tot['spacing'] / pairs,
        'simaa_soprano': {k: v for k, v in sim.items() if k.startswith('SIMAA')},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='v11_scratch.pt')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--bars', type=int, default=16, help='生成多少小节（1 小节 ≈ 4 切片）')
    ap.add_argument('--d', type=int, default=480)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8, help='注意力头数（须与训练时一致）')
    ap.add_argument('--out', type=str, default='data/processed/v11_scratch_eval.json')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bpm', type=int, default=76)
    ap.add_argument('--planner', type=str, default=None,
                    help='和声规划器权重；给了就用"规划器和声"而不是"未知和声"')
    ap.add_argument('--planner-d', type=int, default=256)
    ap.add_argument('--planner-layers', type=int, default=4)
    ap.add_argument('--out-dir', type=str, default='data/generated')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SATBDiffusion(d=args.d, h=args.heads, layers=args.layers, vocab=VOCAB).to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device, weights_only=True))
    model.eval()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    planner = None
    if args.planner:
        planner = HarmonyPlanner(d=args.planner_d, layers=args.planner_layers).to(device)
        planner.load_state_dict(torch.load(ROOT / args.planner, map_location=device,
                                           weights_only=True))
        planner.eval()
        print(f'已载入和声规划器 {args.planner}（和声由规划器生成）')
    T = args.bars * 4
    gens, ends_all, made = [], [], []
    for i in range(args.n):
        cond, ends = build_plan(T, VOCAB, device)
        if planner is not None:
            cond = plan_harmony(planner, cond, T, device, seed=args.seed + i)
        pitch = torch.full((1, N_VOICES, T), VOCAB.rest, dtype=torch.long, device=device)
        rhythm = torch.full((1, N_VOICES, T), HOLD, dtype=torch.long, device=device)
        known = torch.zeros((1, N_VOICES, T), dtype=torch.bool, device=device)   # 全挖
        gp, gr = model.generate(pitch, rhythm, known, cond, steps=1, argmax=False,
                                temp=1.0, seed=args.seed + i)
        gens.append((gp[0].cpu(), gr[0].cpu()))
        ends_all.append(ends)
        offs = [t * 1.0 for t in range(T)]                # 计划：每切片 1 拍（4 切片 = 1 小节）
        f = out_dir / f'v11_scratch_{i + 1}.mid'
        write_midi(f, gp[0].cpu(), gr[0].cpu(), offs, VOCAB, args.bpm)
        made.append(f.name)
        print(f'  第{i + 1}首 → {f.name}')

    ref = corpus_reference()
    res = evaluate_generated(gens, ends_all, ref)
    print('\n=== 音乐性评估（自由生成 vs 真众赞歌）===')
    print(f'{"指标":32s} {"生成":>10s} {"真巴赫":>10s}')
    for k, label in (('key_c_ratio', '落在 C 调(含 c 小调)比例'),
                     ('key_conf_abs_mean', '调性明确程度(平均置信)'),
                     ('key_conf_chorale', 'C 度(有符号, 旧口径)'),
                     ('chord_explainable', '纵向音响可解释为和弦'),
                     ('cadence_bass_on_tonic_or_dom', '乐句末低音在主/属音'),
                     ('parallel_5_per100', '平行五度/100 对'),
                     ('parallel_8_per100', '平行八度/100 对'),
                     ('crossing_per100', '声部交越/100 对'),
                     ('spacing_per100', '间距>八度/100 对')):
        g = res.get(k, float('nan'))
        r = ref.get(k, float('nan'))
        print(f'{label:32s} {g:10.3f} {r:10.3f}')
    json.dump({'config': {**vars(args), 'n_generated': len(gens)},
               'generated': res, 'reference': ref, 'files': made},
              open(ROOT / args.out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'→ {args.out}')


if __name__ == '__main__':
    main()
