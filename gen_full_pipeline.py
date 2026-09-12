"""
完整生成流水线（三级模型）—— 从**曲式模板**到四声部成品，全部内容由模型生成。

    曲式模板（规则：4 句、半终止/全终止位置、长度）
        ↓  ① 旋律模型 (v13_melody, 只训 Soprano, 无和声条件)
    Soprano 旋律
        ↓  ② 和声规划器 (harmony_planner_mel, 以旋律+模板为输入)
    每切片的和弦 (功能/类型/根音)
        ↓  ③ 四声部实现器 (v11/v12 扩散模型, 以旋律+和声为条件)
    Alto / Tenor / Bass

与前几版的区别（哪些是"生成"的）
------------------------------------------------
    gen_satb.py      : 旋律=真值, 和声=真值  → 只生成内声部
    gen_planned.py   : 旋律=真值, 和声=规划器 → 生成和声+内声部
    **本脚本**        : 旋律=旋律模型, 和声=规划器, 声部=实现器 → **全部生成**
    唯一外部输入是曲式模板（长度/乐句/终止式位置），这是作曲上的"形式要求"。

评估：全部配真巴赫基线；另存 MIDI 便于盲听。
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import torch
import torch.nn.functional as F

from gen_satb import write_midi
from gen_from_scratch import (build_plan, chord_explainable, corpus_reference,
                              evaluate_generated, VOCAB)
from model.satb_diffusion import N_VOICES, SATBDiffusion, SatbVocab, voiceleading_metrics
from train_harmony_planner import (N_FUNC_LABEL, N_TYPE_LABEL, HarmonyPlanner,
                                   sample_plan_token)
from train_v10_satb import COND_KEYS, HOLD


@torch.no_grad()
def gen_melody(model, cond, T, device, seed=0, temp=0.9):
    """① 生成 Soprano（全掩码，无和声条件）。"""
    pitch = torch.full((1, 1, T), VOCAB.rest, dtype=torch.long, device=device)
    rhythm = torch.full((1, 1, T), HOLD, dtype=torch.long, device=device)
    known = torch.zeros((1, 1, T), dtype=torch.bool, device=device)
    c = {k: cond[k] for k in COND_KEYS}
    torch.manual_seed(seed)
    gp, gr = model.generate(pitch, rhythm, known, c, steps=1, argmax=False, temp=temp)
    return gp[0], gr[0]


@torch.no_grad()
def plan_harmony(planner, cond, sop, T, device, seed=0, temp=0.9):
    """② 以旋律+模板为条件，自回归生成和弦计划。"""
    torch.manual_seed(seed)
    pf = torch.full((1, 1), -1, dtype=torch.long, device=device)
    pt = torch.full((1, 1), -1, dtype=torch.long, device=device)
    pr = torch.full((1, 1), -1, dtype=torch.long, device=device)
    fs, ts, rs = [], [], []
    for i in range(T):
        c = {k: cond[k][:, i:i + 1] for k in
             ('pos_bin', 'toend_bin', 'phrase_bin', 'cadence', 'style')}
        lf, lt, lr = planner(c, pf, pt, pr, sop=sop[:, i:i + 1])
        fs.append(sample_plan_token(lf[0, -1], N_FUNC_LABEL, temp))
        ts.append(sample_plan_token(lt[0, -1], N_TYPE_LABEL, temp))
        rs.append(sample_plan_token(lr[0, -1], 12, temp))          # root 0-11
        pf = torch.cat([pf, torch.tensor([[fs[-1]]], device=device)], 1)
        pt = torch.cat([pt, torch.tensor([[ts[-1]]], device=device)], 1)
        pr = torch.cat([pr, torch.tensor([[rs[-1]]], device=device)], 1)
    out = dict(cond)
    out['func'] = torch.tensor([fs], device=device)
    out['type'] = torch.tensor([ts], device=device)
    out['root'] = torch.tensor([rs], device=device)
    return out


@torch.no_grad()
def realize(realizer, cond, sop, T, device, seed=0, temp=0.85):
    """③ 以旋律+规划的和声为条件，生成其余三个声部。"""
    pitch = torch.full((1, N_VOICES, T), VOCAB.rest, dtype=torch.long, device=device)
    rhythm = torch.full((1, N_VOICES, T), HOLD, dtype=torch.long, device=device)
    pitch[0, 0] = sop                       # Soprano 作为已知上下文
    known = torch.zeros((1, N_VOICES, T), dtype=torch.bool, device=device)
    known[:, 0] = True
    torch.manual_seed(seed)
    gp, gr = realizer.generate(pitch, rhythm, known, cond, steps=1,
                               argmax=True, temp=temp)
    return gp[0], gr[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--melody', type=str, default='v13_melody.pt')
    ap.add_argument('--melody-d', type=int, default=384)
    ap.add_argument('--melody-layers', type=int, default=6)
    ap.add_argument('--planner', type=str, default='harmony_planner_mel.pt')
    ap.add_argument('--realizer', type=str, default='v11_scratch.pt')
    ap.add_argument('--realizer-d', type=int, default=480)
    ap.add_argument('--realizer-layers', type=int, default=8)
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--bars', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bpm', type=int, default=76)
    ap.add_argument('--out-dir', type=str, default='data/generated/full')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mel = SATBDiffusion(d=args.melody_d, h=8, layers=args.melody_layers, vocab=VOCAB,
                        n_voices=1).to(device)
    mel.load_state_dict(torch.load(ROOT / args.melody, map_location=device, weights_only=True))
    mel.eval()
    planner = HarmonyPlanner(d=256, layers=4).to(device)
    planner.load_state_dict(torch.load(ROOT / args.planner, map_location=device, weights_only=True))
    planner.eval()
    rz = SATBDiffusion(d=args.realizer_d, h=8, layers=args.realizer_layers, vocab=VOCAB).to(device)
    rz.load_state_dict(torch.load(ROOT / args.realizer, map_location=device, weights_only=True))
    rz.eval()
    print(f'三个模型就绪: 旋律 {args.melody} / 和声 {args.planner} / 声部 {args.realizer}')

    T = args.bars * 4
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    gens, ends_all = [], []
    for i in range(args.n):
        cond, ends = build_plan(T, VOCAB, device)
        sop, _sop_r = gen_melody(mel, cond, T, device, seed=args.seed + i)
        cond = plan_harmony(planner, cond, sop, T, device, seed=args.seed + i)
        pitch, rhythm = realize(rz, cond, sop, T, device, seed=args.seed + i)
        gens.append((pitch.cpu(), rhythm.cpu()))
        ends_all.append(ends)
        offs = [float(t) for t in range(T)]
        f = out_dir / f'full_{i + 1}.mid'
        write_midi(f, pitch.cpu(), rhythm.cpu(), offs, VOCAB, args.bpm)
        roots = [int(x) for x in cond['root'][0].cpu().tolist()]
        print(f'  第{i + 1}首 → {f.name} | 和声根音序列(前 16): {roots[:16]}')

    ref = corpus_reference()
    res = evaluate_generated(gens, ends_all, ref)
    print('\n=== 端到端生成（模板→旋律→和声→四声部） vs 真众赞歌 ===')
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
        print(f'{label:32s} {res.get(k, float("nan")):10.3f} {ref.get(k, float("nan")):10.3f}')
    json.dump({'config': {**vars(args), 'n_generated': len(gens)},
               'generated': res, 'reference': ref},
              open(ROOT / 'data/processed/v13_full_eval.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f'→ {out_dir} 与 data/processed/v13_full_eval.json')


if __name__ == '__main__':
    main()
