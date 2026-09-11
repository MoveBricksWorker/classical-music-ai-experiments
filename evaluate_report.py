"""
评估与报告 —— 把全部证据链串起来:

    1. 人类语料参考值 (乐理指标 + 自比 SIMAA/马氏距离)
    2. 扩散模型生成 (给定和声) 的乐理指标 / SIMAA / 马氏距离
    3. 自回归 MelodyGPT 基线 (需权重) 的同口径对比
    4. 输出 data/processed/eval_report.json + eval_report.md

用法:
    python evaluate_report.py [--n 60] [--steps 16] [--temp 1.0] [--remask 6]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

import argparse, json, math, random, time

import numpy as np
import torch

from constants import F2ID_MELODY, T2ID, MELODY_RHYTHM
from model.melody_diffusion import MelodyDiffusion, PITCH_MASK, RHYTHM_MASK, REST
from model.architectures import MelodyGPT
from metrics.theory_metrics import evaluate_pieces
from metrics.anti_imitation import (load_corpus_pieces, simaa, mahalanobis,
                                    melody_features)


# ═══════════════════════════════════════════════════════════════
# 语料切分 (与训练同口径: SEQ=16 窗口)
# ═══════════════════════════════════════════════════════════════

def load_windows(json_path, n=None, split=0.9, with_beat=False, block_conds=False,
                 with_struct=False, with_phrase=False,
                 phrases_path='data/processed/melody_phrases_v1.json'):
    from collections import Counter
    d = json.load(open(json_path, encoding='utf-8'))
    # 拍位: 按乐曲从第 0 拍累计时值 (mod 4), 与训练脚本同口径
    beats = []
    if with_beat:
        t = 0.0
        for note in d:
            if note.get('prev_pc') is None:
                t = 0.0
            beats.append(int(t) % 4)
            t += MELODY_RHYTHM.get(min(7, note['rhythm']), 1.0)
    struct_labels = []
    if with_struct:
        from train_melody_diffusion import compute_struct_labels
        struct_labels = compute_struct_labels(d)
    phrase_labels = []
    if with_phrase:
        from train_melody_diffusion import load_phrase_labels
        phrase_labels = load_phrase_labels(str(ROOT / phrases_path), len(d))
    samples = []
    for i in range(0, len(d) - 16, 4):
        chunk = d[i:i + 16]
        funcs = [F2ID_MELODY.get(c.get('func', 'Other'), 4) for c in chunk]
        types = [T2ID.get(c.get('type', 'M'), 0) for c in chunk]
        roots = [min(11, c['root']) for c in chunk]
        if block_conds:
            bf, bt, br = [], [], []
            for j in range(0, 16, 4):
                bf += [Counter(funcs[j:j + 4]).most_common(1)[0][0]] * 4
                bt += [Counter(types[j:j + 4]).most_common(1)[0][0]] * 4
                br += [Counter(roots[j:j + 4]).most_common(1)[0][0]] * 4
            funcs, types, roots = bf, bt, br
        s = {
            'func': funcs, 'type': types, 'root': roots,
            'pc': [min(REST, c['pc']) for c in chunk],
            'rhythm': [min(7, c['rhythm']) for c in chunk],
        }
        if with_beat:
            s['beat'] = beats[i:i + 16]
        if with_struct:
            # 训练同口径: 按乐曲边界的相对位置
            st = struct_labels[i:i + 16]
            s['pos_bin'] = [x[0] for x in st]
            s['toend_bin'] = [x[1] for x in st]
        if with_phrase:
            s['phrase_bin'] = phrase_labels[i:i + 16]
        samples.append(s)
    random.Random(42).shuffle(samples)
    sp = int(len(samples) * split)
    if n:
        return samples[:n]
    return samples, samples[:sp], samples[sp:]


# ═══════════════════════════════════════════════════════════════
# 生成 (扩散模型, 给定和声)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_diffusion(model, samples, device, steps, temp, remask, scorer=None,
                       with_beat=False, sampler='maskgit'):
    """对每个窗口的和声条件生成旋律, 返回 (pieces, beats_list)。

    sampler: 'maskgit' (置信度硬调度, 快速) 或 'd3pm' (GETMusic 祖先采样)。
    """
    out = []
    beats_list = []
    for s in samples:
        fn = torch.tensor([s['func']], device=device)
        tp = torch.tensor([s['type']], device=device)
        rt = torch.tensor([s['root']], device=device)
        beat = torch.tensor([s['beat']], device=device) if with_beat else None
        pos_bin = torch.tensor([s['pos_bin']], device=device) if 'pos_bin' in s else None
        toend_bin = torch.tensor([s['toend_bin']], device=device) if 'toend_bin' in s else None
        phrase_bin = torch.tensor([s['phrase_bin']], device=device) if 'phrase_bin' in s else None
        if sampler == 'd3pm':
            gp, grh = model.generate_d3pm(fn, tp, rt, steps=steps, temp=temp,
                                          scorer=scorer, scorer_steps=4,
                                          expand=False, beat=beat)
        else:
            gp, grh = model.generate(fn, tp, rt, steps=steps, temp=temp,
                                     remask_steps=remask, remask_ratio=0.3,
                                     scorer=scorer, scorer_steps=4, expand=False,
                                     beat=beat, pos_bin=pos_bin, toend_bin=toend_bin)
        out.append((gp[0].tolist(), grh[0].tolist()))
        if with_beat:
            beats_list.append(s['beat'])
    return out, beats_list


def make_scorer(chord_tones_all):
    def scorer(pc, cts, prev_pc, step):
        score = 0.0
        if pc in cts:
            score += 0.3
        dist = min(abs(pc - prev_pc), 12 - abs(pc - prev_pc))
        if 1 <= dist <= 2:
            score += 0.4
        elif dist >= 6:
            score -= 0.3
        if pc == prev_pc:
            score -= 0.3
        return score
    return scorer


def chord_tones_of(sample):
    t_map = {0: [0, 4, 7], 1: [0, 3, 7], 2: [0, 3, 6], 3: [0, 4, 7, 10],
             4: [0, 4, 8], 5: [0, 3, 7, 10], 6: [0, 3, 6, 9]}
    return [[(sample['root'][i] + iv) % 12 for iv in t_map.get(sample['type'][i], [0, 4, 7])]
            for i in range(16)]


@torch.no_grad()
def generate_baseline(model, samples, device, scorer=None):
    """自回归 MelodyGPT 在同条件 (同和声窗口) 下生成, 供公平对比。

    16 音窗口 = 4 个"和弦" × 4 音, 与扩散路径的展开口径一致。
    """
    out = []
    for s in samples:
        fn = torch.tensor([s['func'][::4]], device=device)
        tp = torch.tensor([s['type'][::4]], device=device)
        rt = torch.tensor([s['root'][::4]], device=device)
        gp, _, grh = model.gen(fn, tp, rt, max_len=16, temp=1.0,
                               notes_per_chord=4, scorer=scorer)
        out.append((gp[0].tolist(), grh[0].tolist()))
    return out


def window_stats(gen_src, gen_pieces):
    """和弦音吻合率 + 振荡比例 (两种模型共用)。"""
    ct_hits = []
    for sample, (pcs, _) in zip(gen_src, gen_pieces):
        cts_all = chord_tones_of(sample)
        hits = sum(1 for i, p in enumerate(pcs) if p < REST and p in cts_all[i])
        valid = sum(1 for p in pcs if p < REST)
        ct_hits.append(hits / valid if valid else 0.0)
    osc = []
    for sample, (pcs, _) in zip(gen_src, gen_pieces):
        cts_all = chord_tones_of(sample)
        cnt = 0
        for i in range(len(pcs) - 3):
            w = [p for p in pcs[i:i + 4] if p < REST]
            if len(w) == 4 and len(set(w)) <= 2 and all(p in cts_all[i] for p in w):
                cnt += 1
        osc.append(cnt / max(len(pcs) - 3, 1))
    return {
        'chord_tone_ratio': {'mean': float(np.mean(ct_hits)), 'std': float(np.std(ct_hits))},
        'oscillation_ratio': {'mean': float(np.mean(osc)), 'std': float(np.std(osc))},
    }


# ═══════════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--remask', type=int, default=6)
    parser.add_argument('--sampler', choices=['maskgit', 'd3pm'], default='maskgit',
                        help='maskgit=置信度硬调度; d3pm=GETMusic 祖先采样 (steps=100)')
    parser.add_argument('--scorer', action='store_true', default=True)
    parser.add_argument('--no-scorer', dest='scorer', action='store_false')
    parser.add_argument('--pt', type=str, default='melody_diffusion_v5_blockconds.pt')
    parser.add_argument('--d', type=int, default=288, help='模型宽度 (与权重匹配)')
    parser.add_argument('--layers', type=int, default=8, help='模型层数 (与权重匹配)')
    parser.add_argument('--beat', action='store_true', help='权重含拍位条件流')
    parser.add_argument('--no-block-conds', dest='block_conds', action='store_false',
                        default=True,
                        help='关闭 4 音块恒定条件 (v1 老模型口径)')
    parser.add_argument('--struct', action='store_true', help='权重含结构流')
    parser.add_argument('--phrase', action='store_true', help='权重含乐句流')
    parser.add_argument('--rope', action='store_true', help='权重用 RoPE')
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--baseline-pt', type=str, default='melody_model_cloud_v4.pt',
                        help='自回归 MelodyGPT 权重 (存在则做同条件对比)')
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    melody_json = ROOT / 'data/processed/melody_llm_full_v2.json'
    t0 = time.time()

    # 1. 语料参考
    pieces = load_corpus_pieces(str(melody_json))
    ref_theory = evaluate_pieces(pieces)
    flat_train = [p for pcs, _ in pieces for p in pcs]
    ref_simaa = simaa(flat_train, [pcs[:16] for pcs, _ in pieces if len(pcs) >= 16])
    ref_mahal = mahalanobis(pieces, pieces)

    report = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': vars(args),
        'device': device,
        'corpus': {
            'n_pieces': len(pieces),
            'theory': ref_theory,
            'simaa_self': ref_simaa,
            'mahalanobis_self': ref_mahal,
        },
        'diffusion': {},
    }

    # 2. 扩散模型生成
    print(f'加载扩散模型 {args.pt} (d={args.d}, L={args.layers}, beat={args.beat}, '
          f'struct={args.struct}, phrase={args.phrase}, rope={args.rope}) ...')
    model = MelodyDiffusion(d=args.d, h=args.heads, L=args.layers, max_len=256,
                            num_funcs=len(F2ID_MELODY), num_types=len(T2ID),
                            num_beat_positions=4 if args.beat else None,
                            num_pos_bins=8 if args.struct else 0,
                            num_toend_bins=4 if args.struct else 0,
                            num_phrase_bins=6 if args.phrase else 0,
                            use_rope=args.rope).to(device)
    model.load_state_dict(torch.load(ROOT / args.pt, map_location=device, weights_only=True))
    model.eval()

    _, train_win, val_win = load_windows(str(melody_json), with_beat=args.beat,
                                         block_conds=args.block_conds,
                                         with_struct=args.struct,
                                         with_phrase=args.phrase)
    gen_src = val_win[:args.n]
    scorer = None
    if args.scorer:
        scorer = make_scorer(chord_tones_of(gen_src[0]))

    print(f'生成 {len(gen_src)} 条旋律 (steps={args.steps}, temp={args.temp}, remask={args.remask}) ...')
    t1 = time.time()
    gen_pieces, gen_beats = generate_diffusion(model, gen_src, device, args.steps,
                                               args.temp, args.remask,
                                               scorer=scorer, with_beat=args.beat)
    gen_time = time.time() - t1

    # 拍位模型: 节奏指标与条件同框架 (shift = -窗口首音拍位)
    gen_theory_in = [(p, r, b) for (p, r), b in zip(gen_pieces, gen_beats)] \
        if args.beat else gen_pieces
    gen_theory = evaluate_pieces(gen_theory_in)
    gen_tokens = [p for p, _ in gen_pieces]
    gen_simaa = simaa(flat_train, gen_tokens)
    gen_mahal = mahalanobis(pieces, gen_pieces)

    report['diffusion'] = {
        'n_generated': len(gen_pieces),
        'gen_sec_total': gen_time,
        'gen_sec_per_piece': gen_time / len(gen_pieces),
        'theory': gen_theory,
        'simaa': gen_simaa,
        'mahalanobis': gen_mahal,
        **window_stats(gen_src, gen_pieces),
    }

    # 3. 自回归基线 (同条件对比)
    baseline_pt = ROOT / args.baseline_pt
    if baseline_pt.exists():
        print(f'加载自回归基线 {args.baseline_pt} ...')
        base_model = MelodyGPT(d=504, h=8, L=16,
                               num_funcs=len(F2ID_MELODY), num_types=len(T2ID),
                               num_roles=12, num_rhythm=8, pitch_classes=14).to(device)
        base_model.load_state_dict(
            torch.load(baseline_pt, map_location=device, weights_only=True), strict=True)
        base_model.eval()
        base_pieces = generate_baseline(base_model, gen_src, device, scorer=scorer)
        base_tokens = [p for p, _ in base_pieces]
        report['autoregressive_baseline'] = {
            'n_generated': len(base_pieces),
            'theory': evaluate_pieces(base_pieces),
            'simaa': simaa(flat_train, base_tokens),
            'mahalanobis': mahalanobis(pieces, base_pieces),
            **window_stats(gen_src, base_pieces),
        }
    else:
        print(f'跳过基线 (未找到 {args.baseline_pt})')
        report['autoregressive_baseline'] = None

    # 3. 保存生成样例 (JSON + 简单 MIDI, 附和声条件)
    out_dir = ROOT / 'data/generated'
    out_dir.mkdir(exist_ok=True)
    samples_out = []
    for s, (p, r) in zip(gen_src, gen_pieces):
        samples_out.append({
            'pitch': p, 'rhythm': r,
            'harmony': [{'root': s['root'][j * 4], 'type_id': s['type'][j * 4]}
                        for j in range(4)],
        })
    json.dump({'samples': samples_out},
              open(out_dir / 'diffusion_samples.json', 'w'), ensure_ascii=False, indent=2)

    # 4. 写报告
    json.dump(report, open(ROOT / 'data/processed/eval_report.json', 'w'),
              ensure_ascii=False, indent=2)
    write_markdown(report, ROOT / 'data/processed/eval_report.md')
    print(f'完成 (总耗时 {time.time()-t0:.0f}s) → data/processed/eval_report.json / .md')


def write_markdown(rep, path):
    c, d = rep['corpus'], rep['diffusion']
    b = rep.get('autoregressive_baseline')
    th_ref, th_gen = c['theory'], d['theory']

    def row(name, ref, gen, base, fmt='{:.3f}'):
        def cell(x):
            return fmt.format(x['mean']) + ' ± ' + fmt.format(x['std'])
        line = f'| {name} | {cell(ref)} | {cell(gen)} |'
        if base is not None:
            line += f' {cell(base)} |'
        return line

    head = '| 指标 | 人类语料 | 扩散模型 |'
    sep = '|------|----------|----------|'
    if b is not None:
        head += ' MelodyGPT |'
        sep += '-----------|'
    lines = [
        '# 旋律生成评估报告 (扩散非自回归 vs 自回归基线)',
        '',
        f'- 生成时间: {rep["generated_at"]} | 设备: {rep["device"]}',
        f'- 配置: steps={rep["config"]["steps"]}, temp={rep["config"]["temp"]}, remask={rep["config"]["remask"]}, scorer={rep["config"]["scorer"]}',
        f'- 扩散生成样本: {d["n_generated"]} 条, 平均 {d["gen_sec_per_piece"]:.2f}s/条',
        '',
        '## 1. 乐理合格度 (MusicAIR 口径)',
        '',
        head, sep,
        row('key confidence', th_ref['key'], th_gen['key'], b and b['theory']['key']),
        row('avg interval (半音)', th_ref['smoothness']['avg_interval'], th_gen['smoothness']['avg_interval'], b and b['theory']['smoothness']['avg_interval']),
        row('step ratio', th_ref['smoothness']['step_ratio'], th_gen['smoothness']['step_ratio'], b and b['theory']['smoothness']['step_ratio']),
        row('direction change rate', th_ref['smoothness']['direction_change_rate'], th_gen['smoothness']['direction_change_rate'], b and b['theory']['smoothness']['direction_change_rate']),
        row('strong onset ratio', th_ref['rhythm']['strong_onset_ratio'], th_gen['rhythm']['strong_onset_ratio'], b and b['theory']['rhythm']['strong_onset_ratio']),
        row('downbeat onset ratio', th_ref['rhythm']['downbeat_onset_ratio'], th_gen['rhythm']['downbeat_onset_ratio'], b and b['theory']['rhythm']['downbeat_onset_ratio']),
        row('long-on-strong ratio', th_ref['rhythm']['long_on_strong_ratio'], th_gen['rhythm']['long_on_strong_ratio'], b and b['theory']['rhythm']['long_on_strong_ratio']),
        '',
        f'- 和弦音吻合率: 扩散 {d["chord_tone_ratio"]["mean"]:.3f} ± {d["chord_tone_ratio"]["std"]:.3f}'
        + (f' | 基线 {b["chord_tone_ratio"]["mean"]:.3f} ± {b["chord_tone_ratio"]["std"]:.3f}' if b else ''),
        f'- 振荡比例 (4音窗≤2音高且全和弦音): 扩散 {d["oscillation_ratio"]["mean"]:.3f} ± {d["oscillation_ratio"]["std"]:.3f}'
        + (f' | 基线 {b["oscillation_ratio"]["mean"]:.3f} ± {b["oscillation_ratio"]["std"]:.3f}' if b else ''),
        '',
        '## 2. 防模仿证据 (MusicLDM / DRMW 口径)',
        '',
        '### SIMAA (三元组 Jaccard 对训练窗口的最大相似度)',
        '',
        f'- 训练集自比: SIMAA@90={c["simaa_self"]["SIMAA@90"]:.3f}, SIMAA@95={c["simaa_self"]["SIMAA@95"]:.3f}',
        f'- 扩散生成: SIMAA@90={d["simaa"]["SIMAA@90"]:.3f}, SIMAA@95={d["simaa"]["SIMAA@95"]:.3f}, mean_max_sim={d["simaa"]["mean_max_sim"]:.3f}, 记忆率={d["simaa"]["mean_trigram_memory"]:.3f}',
    ]
    if b:
        lines.append(
            f'- 自回归基线: SIMAA@90={b["simaa"]["SIMAA@90"]:.3f}, SIMAA@95={b["simaa"]["SIMAA@95"]:.3f}, mean_max_sim={b["simaa"]["mean_max_sim"]:.3f}, 记忆率={b["simaa"]["mean_trigram_memory"]:.3f}')
    lines += [
        '',
        '### 马氏距离 (17 维符号特征)',
        '',
        f'- 训练分布: {c["mahalanobis_self"]["train_mean"]:.2f} ± {c["mahalanobis_self"]["train_std"]:.2f}',
        f'- 扩散生成: {d["mahalanobis"]["gen_mean"]:.2f} ± {d["mahalanobis"]["gen_std"]:.2f}, Welch t={d["mahalanobis"]["t_stat"]:.2f}, p={d["mahalanobis"]["p_value"]:.3f}, 95分位内={d["mahalanobis"]["frac_within_train95"]:.3f}',
    ]
    if b:
        lines.append(
            f'- 自回归基线: {b["mahalanobis"]["gen_mean"]:.2f} ± {b["mahalanobis"]["gen_std"]:.2f}, Welch t={b["mahalanobis"]["t_stat"]:.2f}, p={b["mahalanobis"]["p_value"]:.3f}, 95分位内={b["mahalanobis"]["frac_within_train95"]:.3f}')
    lines += [
        '',
        '> DRMW 口径: ≤6 连贯 / 3-4 接近风格 / ≥10 随机。',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
