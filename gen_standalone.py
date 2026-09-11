"""
独立生成器 —— 完全不依赖 GitHub 权重 (chord_model_cloud_v4.pt)。

和声: 规则化功能语法 (T-PD-D-T 骨架 + 次功能变化 + 终止式收束, C大调)
旋律: 本地训练的 MelodyDiffusion (melody_diffusion_v1.pt, 6.2M)
      + 乐理 scorer 引导 + 3 候选择优
织体: 阿尔贝蒂/破碎八度/琶音 (复用 gen_full)
输出: data/processed/standalone_full.mid (三轨)

用法: python gen_standalone.py [--chords 48] [--seed N] [--out PATH]
"""
import sys, os, json, random, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'processed')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import torch

from gen_full import (load_diffusion_model, generate_midi, polish_melody,
                      score_candidate, compute_phrase_plan, DIFFUSION_PT)
from constants import CHORD_INTERVALS, F2ID_MELODY, T2ID, ID2DUR
from model.architectures import chord_name_to_info
from motif_form import (phrase_spans_from_labels, plan_form,
                        apply_motif_reuse, apply_registers)

# 功能 → 合法和弦实现 (C大调)
FUNC_CHORDS = {
    'T':  [('I', 3.0), ('I6', 2.0), ('vi', 1.0)],
    'PD': [('IV', 2.0), ('ii', 2.0), ('ii6', 1.0)],
    'D':  [('V', 2.0), ('V7', 3.0), ('vii°', 1.0)],
    'Sec': [('iii', 1.5), ('vi', 1.5), ('IV6', 1.0)],
}
# 乐句骨架 (每乐句 4 和弦的功能序列), 权重控制出现频率
PHRASE_PATTERNS = [
    (['T', 'PD', 'D', 'T'], 3.0),
    (['T', 'T', 'PD', 'D'], 1.0),
    (['T', 'Sec', 'PD', 'D'], 1.5),
    (['PD', 'PD', 'D', 'T'], 1.0),
    (['T', 'PD', 'D', 'D'], 0.5),
]


def load_corpus_progressions(path: str = None) -> list[list[dict]]:
    """加载语料的真实和弦进行 (461 首, 罗马数字级数, 天然适配 C 大调)。"""
    if path is None:
        path = os.path.join(DATA, 'long_sequences_v4.json')
    seqs = json.load(open(path, encoding='utf-8'))
    return [[dict(c) for c in s['chords']] for s in seqs]


def sample_corpus_progression(bank: list[list[dict]], n_chords: int,
                              rng) -> list[dict]:
    """从真实进行库采样拼接成 n_chords 个和弦的骨架。

    - 每段随机取一首曲子的一段 (优先从 T 功能或曲首开始);
    - 时值映射到管线支持的 {2,4} 拍 (真实和声节奏多为 0.5-1 拍, 直接
      使用会超出旋律模型 4 音/和弦的粒度设计);
    - 末段尽量结束在终止式 (乐句完整)。
    """
    out = []
    guard = 0
    while len(out) < n_chords and guard < 20:
        guard += 1
        prog = rng.choice(bank)
        need = n_chords - len(out)
        # 段起点: 从前 1/3 里找一个 T 功能起拍
        starts = [i for i in range(min(len(prog), max(1, len(prog) // 3)))
                  if prog[i].get('func') == 'T'] or [0]
        st = rng.choice(starts)
        seg = prog[st:st + need]
        if not seg:
            continue
        # 时值映射: <1 拍 → 2 拍 (ID 5), >=1 拍 → 4 拍 (ID 7)
        for c in seg:
            dur_beats = ID2DUR.get(c.get('dur', 3), 1.0)
            c['dur'] = 7 if dur_beats >= 1.0 else 5
            c['cad'] = 0
        out.extend(seg)
    return out[:n_chords]


def weighted_choice(weighted_items, rng):
    items = [i for i, _ in weighted_items]
    weights = [w for _, w in weighted_items]
    return rng.choices(items, weights=weights, k=1)[0]
def generate_chords_rule(n_chords: int, seed: int | None = None) -> list[dict]:
    """规则化功能进行: 乐句 = 骨架模式 + 合法和弦实现 + 时值/转位。"""
    rng = random.Random(seed)
    n_phrases = max(2, n_chords // 4)
    phrases = []
    for p in range(n_phrases):
        funcs = weighted_choice(PHRASE_PATTERNS, rng)
        # 最后一个乐句强制终止式 PD→D→T 收束
        if p == n_phrases - 1:
            funcs = ['T', 'PD', 'D', 'T']
        for bi, f in enumerate(funcs):
            chord_name = weighted_choice(FUNC_CHORDS[f], rng)
            # 时值: DUR2ID 量化 ID (5=2拍, 6=3拍, 7=4拍)。
            # 至少 2 拍: 每和弦 4 个旋律音落在八分音符网格, 避免 16 分音符
            # 快速跑动的机械感 (1拍和弦 → slot=0.25拍 = 16分)。
            if bi == 3:
                dur = rng.choice([6, 7])          # 乐句末: 3-4 拍长音
            elif bi == 2:
                dur = rng.choice([5, 6])          # 属功能: 2-3 拍
            else:
                dur = rng.choice([5, 5, 6])
            inv = 0
            if chord_name.endswith('6') or (f == 'T' and rng.random() < 0.3):
                inv = 1
            phrases.append({
                'func': f, 'chord': chord_name, 'dur': dur,
                'beat': bi + 1, 'inv': inv, 'cad': int(bi == 3),
            })
    return phrases[:n_chords]


# ═══════════════════════════════════════════════════════════════
# 主流程 (扩散旋律, 与 gen_full 同口径)
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chords', type=int, default=48)
    parser.add_argument('--harmony', choices=['corpus', 'rule'], default='corpus',
                        help='corpus=采样真实语料进行 (推荐); rule=规则生成')
    parser.add_argument('--form', choices=['llm', 'template', 'off'], default='template',
                        help='曲式层: llm=本地LLM设计, template=模板, off=关闭')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--out', type=str, default=os.path.join(DATA, 'standalone_full.mid'))
    parser.add_argument('--bpm', type=int, default=92)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device} | 扩散权重: {os.path.basename(DIFFUSION_PT)}')

    import random as _r
    rng = _r.Random(args.seed)
    if args.harmony == 'corpus':
        bank = load_corpus_progressions()
        chords = sample_corpus_progression(bank, args.chords, rng)
    else:
        chords = generate_chords_rule(args.chords, args.seed)
    n_chords = len(chords)
    func_dist = {f: sum(1 for c in chords if c['func'] == f) for f in ('T', 'PD', 'D', 'Sec')}
    print(f'规则和声: {n_chords} 和弦, 功能分布 {func_dist}')

    model = load_diffusion_model(device)

    TYPE_REMAP = {'dom7': 'M', 'm7': 'm', 'M7': 'M',
                  'dim7': 'dim', 'hdim7': 'dim', 'aug': 'M'}
    cf_t, ct_t, cr_t = [], [], []
    chord_tones_all = []
    for c in chords:
        r, ct = chord_name_to_info(c['chord'])
        ct = TYPE_REMAP.get(ct, ct)
        fid = min(F2ID_MELODY.get(c['func'], 4), len(F2ID_MELODY) - 1)
        tid = min(T2ID.get(ct, 0), len(T2ID) - 1)
        cf_t += [fid] * 4
        ct_t += [tid] * 4
        cr_t += [r] * 4
        chord_tones_all.append([(r + iv) % 12 for iv in CHORD_INTERVALS.get(ct, [0, 4, 7])])
    cf = torch.tensor([cf_t], device=device)
    ct = torch.tensor([ct_t], device=device)
    cr = torch.tensor([cr_t], device=device)

    def melody_scorer(pc, cts, prev_pc, step):
        s = 0.0
        if pc in cts:
            s += 0.3
        dist = min(abs(pc - prev_pc), 12 - abs(pc - prev_pc))
        if 1 <= dist <= 2:
            s += 0.4
        elif dist >= 6:
            s -= 0.3
        if pc == prev_pc:
            s -= 0.9           # 强重复惩罚 (连续同音呆板)
        if step % 4 == 0:
            s += 0.2 if pc in cts else -0.2
        return s

    # 结构流标签 (与训练同口径: 单曲从头到尾)
    L_gen = n_chords * 4
    struct_pos = torch.tensor([[min(7, int(i / L_gen * 8)) for i in range(L_gen)]],
                              device=device)
    tb = []
    for i in range(L_gen):
        rem = 1.0 - (i + 1) / L_gen
        tb.append(0 if rem < 0.06 else (1 if rem < 0.15 else (2 if rem < 0.40 else 3)))
    struct_toend = torch.tensor([tb], device=device)
    phrase_bin = torch.tensor([compute_phrase_plan(chords)], device=device)

    best, best_score = None, -999
    for cand in range(3):
        gp, grh = model.generate(cf[:, ::4], ct[:, ::4], cr[:, ::4],
                                 steps=16, temp=0.85, remask_steps=6,
                                 remask_ratio=0.3, scorer=melody_scorer,
                                 scorer_steps=4, repeat_damp=0.05,
                                 pos_bin=struct_pos, toend_bin=struct_toend,
                                 phrase_bin=phrase_bin,
                                 seed=(args.seed or 0) * 1000 + cand)
        pcs = [int(p) for p in gp[0].tolist()]
        rhs = [int(r) for r in grh[0].tolist()]
        s = score_candidate(pcs, chord_tones_all)
        if s > best_score:
            best_score, best = s, (pcs, rhs)
    print(f'扩散旋律: 候选最优得分 {best_score:.2f}')

    m_pcs, m_rhythms = best

    # 曲式层 (参考 Guided Musical Form): 动机复用 (A-B-A') + 音区对比
    form_plan = None
    spans = None
    if args.form != 'off':
        labels_ph = compute_phrase_plan(chords)
        spans = phrase_spans_from_labels(labels_ph)
        if len(spans) >= 3:
            form_plan = plan_form(len(spans), rng, use_llm=(args.form == 'llm'))
            n_reuse = sum(1 for p in form_plan if p['reuse'] is not None)
            desc = ' '.join(
                f"{i+1}:{p['register']}" + (f"<-{p['reuse']+1}" if p['reuse'] is not None else '')
                for i, p in enumerate(form_plan))
            print(f'曲式: {len(spans)} 乐句, 动机复用 {n_reuse} 处 | {desc}')

            def _cts(c):
                r, ct = chord_name_to_info(c['chord'])
                return [(r + iv) % 12 for iv in CHORD_INTERVALS.get(ct, [0, 4, 7])]

            m_pcs, m_rhythms = apply_motif_reuse(
                model, m_pcs, m_rhythms, chords, spans, form_plan, _cts,
                keep_ratio=0.6, rng=rng, device=device)

    # 后处理: 跳进最小化八度分配 + 结尾解决 (polish_melody 已重写)
    midi_pitches, m_rhythms = polish_melody(m_pcs, m_rhythms, chords, chord_tones_all)
    if form_plan is not None and spans is not None:
        midi_pitches = apply_registers(midi_pitches, spans, form_plan)
    generate_midi(chords, midi_pitches, m_rhythms, bpm=args.bpm, output_path=args.out)
    print(f'完成 → {args.out}')


if __name__ == '__main__':
    main()
