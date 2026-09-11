"""
句子级生成器 —— v8 模型 (众赞歌训练) + 终止式计划。

流程:
    1. 从众赞歌数据集采样一首作为**和声框架** (块恒定条件, 与训练一致);
    2. 句子计划: 用该曲真实乐句结构 (phrase_toend + 终止式类型)
       —— 两级生成的"规划器"来自理论标注;
    3. v8 扩散生成旋律 (3 候选 + 评分择优);
    4. 渲染: 跳进最小化八度 + 终止式结尾 + 句末气口 → 双轨 MIDI
       (旋律 + 和声垫)。

用法: python gen_sentence.py [--seed N] [--pieces 2] [--out PATH]
"""
import sys, os, json, argparse, random, math

from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import torch
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from constants import CHORD_INTERVALS, F2ID_MELODY, T2ID, MELODY_RHYTHM
from gen_full import alberti_bass, broken_octave, arpeggio, TEXTURES
from model.melody_diffusion import MelodyDiffusion, REST
from chorale_conditions import piece_conditions  # 条件编码唯一实现 (训练同源)

TPB = 480
OCT_BASE = 64        # 旋律八度基准 (E4), 原 72=C5 偏高
CADENCE_NAMES = {0: '弱分句', 1: '全终止', 2: '半终止', 3: '阻碍终止', 4: '变格终止'}
ID2TYPE = {v: k for k, v in T2ID.items()}
ID2FUNC = {v: k for k, v in F2ID_MELODY.items()}


def default_model_path():
    """主模型: 优先 v9 (按曲分组划分训练), 回退 v8 (旧权重, 有泄漏口径)。"""
    for name in ('melody_diffusion_v9_chorales.pt', 'melody_diffusion_v8_chorales.pt'):
        p = ROOT / name
        if p.exists():
            return p
    return ROOT / 'melody_diffusion_v8_chorales.pt'


def load_model(device='cuda', path=None):
    m = MelodyDiffusion(d=320, h=8, L=6, max_len=256,
                        num_funcs=len(F2ID_MELODY), num_types=len(T2ID),
                        num_pos_bins=8, num_toend_bins=4, num_phrase_bins=7,
                        num_cadence_types=5, use_rope=True).to(device)
    m.load_state_dict(torch.load(path or default_model_path(),
                                 map_location=device, weights_only=True))
    m.eval()
    return m


def render_melody_timing(rhythms, is_phrase_end, beats_per_bar=4.0, breath=0.25,
                         grid=0.25):
    """把模型的节奏 bin 渲染成 (起点拍, 时值拍) —— 保证不重叠、不偏格。

    为什么不是"每音一拍"：旧口径 min(bin, 1.0) 把所有音压到 ≤1 拍, 模型
    学到的句末拉长 (bin 6-7) 根本进不了 MIDI。为什么不是自由累计时值：
    会和按小节锁定的和声/织体错位 (此前修过的"拍号对不上")。
    做法: 小节仍锁 4 拍(4 音), 小节内按节奏 bin 比例分配, 再吸附到 `grid`
    (默认十六分网格, 减少合并); 时值取"相邻起点之差", 因此天然不重叠; 句末音留
    `breath` 拍气口。相邻音被吸到同一格时后者并入前者 (避免零时值)。
    """
    n = len(rhythms)
    # 1) 小节内按比例分配 (拍)
    frac = [0.0] * n
    bar_end = {}
    for b0 in range(0, n, 4):
        bar_start = (b0 // 4) * beats_per_bar
        idxs = list(range(b0, min(b0 + 4, n)))
        w = [MELODY_RHYTHM.get(rhythms[i], 0.5) for i in idxs]
        total = sum(w) or 1.0
        t = 0.0
        for k, i in enumerate(idxs):
            frac[i] = bar_start + t
            t += beats_per_bar * w[k] / total
        bar_end[b0] = bar_start + beats_per_bar
    # 2) 吸附网格 + 单调化
    q = []
    last_valid = None
    for i in range(n):
        v = round(frac[i] / grid) * grid
        if last_valid is not None and v <= last_valid:
            v = None                      # 与前音同格 → 并入前音
        else:
            last_valid = v
        q.append(v)
    # 3) 时值 = 下一有效起点 (或小节末) - 本音起点
    out = []
    for i in range(n):
        if q[i] is None:
            continue
        nxt = None
        for j in range(i + 1, n):
            if q[j] is not None:
                nxt = q[j] if (j // 4) == (i // 4) else bar_end[(i // 4) * 4]
                break
        if nxt is None:
            nxt = bar_end[(i // 4) * 4]
        dur = nxt - q[i]
        if is_phrase_end[i]:
            dur = max(dur - breath, grid)
        out.append((i, q[i], dur))
    return out


def assign_octaves(pcs, base=OCT_BASE, lo=55, hi=79):
    out, prev = [], None
    for pc in pcs:
        if pc >= REST:
            out.append(-1)
            continue
        best, bd = None, 1e9
        for o in range(2, 7):
            m = pc + o * 12
            if m < lo or m > hi:
                continue
            d = abs(m - base) if prev is None else abs(m - prev)
            if d < bd:
                bd, best = d, m
        out.append(best if best is not None else pc + 60)
        prev = best if best is not None else prev
    return out


def resolve_cadence_endings(pitches, cond, target_pc_of_arrival):
    """按终止式类型修正各到达点收束音 (全终止→和弦根音; 半终止→根音/五音)。"""
    out = list(pitches)
    n = len(out)
    for i in range(n):
        if cond['phrase_bin'][i] != 0 or out[i] < 0:
            continue
        if i == n - 1:
            continue
        cad = cond['cadence'][i]
        tgt = target_pc_of_arrival[i]
        if cad == 1 or cad == 4:      # 全/变格 → 主音 (和弦根音)
            cands = [tgt + o * 12 for o in range(2, 7)]
            prev = out[i - 1] if i > 0 and out[i - 1] >= 0 else 72
            out[i] = min(cands, key=lambda m: abs(m - prev))
    # 末音: 一律落主音 C (框架均已移调到 C, 终曲必须收在主音)
    last = max((j for j in range(n) if out[j] >= 0), default=None)
    if last is not None:
        prev = out[last - 1] if last > 0 and out[last - 1] >= 0 else 72
        cands = [0 + o * 12 for o in range(2, 7)]
        out[last] = min(cands, key=lambda m: abs(m - prev))
    return out


def write_midi(chords_span, midi_pitches, rhythms, out_path, bpm=76,
               breath=0.25, render='rhythm'):
    """旋律 + 织体 + 和声垫 三轨 MIDI。

    render='rhythm' (默认): 小节内按模型节奏 bin 比例分配 4 拍 (句末留气口);
    render='slot'  (旧口径): 每音固定 1 拍槽位, 时值截断到 1 拍 —— 
    会把模型学到的句末拉长压掉, 仅用于复现历史成品。
    """
    mid = MidiFile(ticks_per_beat=TPB)
    t0 = MidiTrack()
    t0.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    t0.append(MetaMessage('time_signature', numerator=4, denominator=4))
    mid.tracks.append(t0)

    # 旋律: 每 4 音一小节; 句末气口
    mel_ev = []
    n = len(midi_pitches)
    if render == 'rhythm':
        is_end = [bool(i < len(chords_span) and chords_span[i]['phrase_bin'] == 0)
                  for i in range(n)]
        for i, onset, dur in render_melody_timing(rhythms, is_end, breath=breath):
            pc = midi_pitches[i]
            if pc < 0:
                continue
            t_on = int(round(onset * TPB))
            t_off = t_on + max(int(round(dur * TPB)), 1)
            mel_ev.append((t_on, 'on', pc)); mel_ev.append((t_off, 'off', pc))
    else:
        for i, (pc, rh) in enumerate(zip(midi_pitches, rhythms)):
            bar, j = i // 4, i % 4
            note_beats = min(MELODY_RHYTHM.get(rh, 0.5), 1.0)
            if chords_span and chords_span[i]['phrase_bin'] == 0 and i != n - 1:
                note_beats *= 0.65
            t_on = bar * 4 * TPB + j * TPB
            t_off = t_on + max(int(round(note_beats * TPB)), 1)
            if pc >= 0:
                mel_ev.append((t_on, 'on', pc)); mel_ev.append((t_off, 'off', pc))
    mel = MidiTrack(); mel.append(Message('program_change', program=0, channel=0, time=0))
    last = 0
    mel_ev.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
    for t, tp, p in mel_ev:
        mel.append(Message('note_on' if tp == 'on' else 'note_off', note=p,
                           velocity=72 if tp == 'on' else 0, channel=0, time=t - last))
        last = t
    mel.append(MetaMessage('end_of_track'))
    mid.tracks.append(mel)

    # 织体轨 (左手): 阿尔贝蒂低音 / 破碎八度 / 琶音, 每小节 8 个八分音符
    tex_ev = []
    rng_t = random.Random(7)
    for bar, ch in enumerate(chords_span[::4]):
        ct = ID2TYPE.get(ch['type'], 'M')
        tex_func = rng_t.choice(TEXTURES)
        if tex_func is arpeggio:
            notes_tex = tex_func(ch['root'], ct, n=8, start_octave=3)
        else:
            notes_tex = tex_func(ch['root'], ct, n=8, octave_low=3)
        t0 = bar * 4 * TPB
        for j, p in enumerate(notes_tex):
            step = 4 * TPB // 8
            tex_ev.append((t0 + j * step, 'on', p))
            tex_ev.append((t0 + (j + 1) * step, 'off', p))
    tex = MidiTrack(); tex.append(Message('program_change', program=0, channel=1, time=0))
    tex_ev.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
    last = 0
    for t, tp, p in tex_ev:
        tex.append(Message('note_on' if tp == 'on' else 'note_off', note=p,
                           velocity=50 if tp == 'on' else 0, channel=1, time=t - last))
        last = t
    tex.append(MetaMessage('end_of_track'))
    mid.tracks.append(tex)

    # 和弦垫: 每 4 音一小节 (中音区, 轻)
    pad = MidiTrack(); pad.append(Message('program_change', program=0, channel=2, time=0))
    last = 0
    pad_ev = []
    for bar, ch in enumerate(chords_span[::4]):
        ivs = sorted(CHORD_INTERVALS.get(ID2TYPE.get(ch['type'], 'M'), [0, 4, 7]))
        t_on = bar * 4 * TPB
        t_off = t_on + 4 * TPB
        for iv in ivs:
            pad_ev.append((t_on, 'on', ch['root'] + 48 + iv))
            pad_ev.append((t_off, 'off', ch['root'] + 48 + iv))
    pad_ev.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
    for t, tp, p in pad_ev:
        pad.append(Message('note_on' if tp == 'on' else 'note_off', note=p,
                           velocity=46 if tp == 'on' else 0, channel=2, time=t - last))
        last = t
    pad.append(MetaMessage('end_of_track'))
    mid.tracks.append(pad)
    mid.save(str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--pieces', type=int, default=1,
                        help='生成几首 (依次取不同和声框架, 输出加 _1.._N 后缀)')
    parser.add_argument('--out', type=str, default='data/generated/sentence_piece.mid')
    parser.add_argument('--model', type=str, default=None,
                        help='模型权重路径 (默认 melody_diffusion_v8_chorales.pt)')
    parser.add_argument('--bpm', type=int, default=76)
    parser.add_argument('--candidates', type=int, default=3)
    parser.add_argument('--render', choices=['rhythm', 'slot'], default='rhythm',
                        help='rhythm=小节内按节奏比例分配 (默认); slot=旧口径每音 1 拍')
    parser.add_argument('--breath', type=float, default=0.25, help='句末气口 (拍)')
    parser.add_argument('--mode', choices=['any', 'major', 'minor'], default='any',
                        help='框架调式过滤 (major=只要大调众赞歌)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(args.seed)
    model = load_model(device, args.model)

    data = json.load(open(ROOT / 'data/processed/chorales_sentences_v1.json', encoding='utf-8'))
    pool = [p for p in data if len(p['notes']) >= 32]
    if args.mode != 'any':
        pool = [p for p in pool if p['key_original'].split()[-1] == args.mode]
    if not pool:
        print(f'没有 {args.mode} 调框架可用'); return

    out_base = ROOT / args.out if not os.path.isabs(args.out) else Path(args.out)
    used = set()
    for pi in range(args.pieces):
        cands = [p for p in pool if p['id'] not in used]
        if not cands:
            break
        piece = rng.choice(cands)
        used.add(piece['id'])
        out = out_base if args.pieces == 1 else out_base.with_name(
            f'{out_base.stem}_{pi + 1}{out_base.suffix}')
        generate_piece(model, piece, rng, device, args, out)
        print(f'→ {out}')


def generate_piece(model, piece, rng, device, args, out):
    cond = piece_conditions(piece)
    n = len(piece['notes'])
    print(f"和声框架: {piece['title'][:40] or piece['id']} ({n} 音, "
          f"{len(piece['phrases'])} 乐句, 原调 {piece['key_original']})")
    cad_seq = ' '.join(CADENCE_NAMES[cond['cadence'][i]] for i in sorted(
        {c['note'] for c in piece['phrases']}))
    print(f'终止式计划: {cad_seq}')

    fn = torch.tensor([cond['func']], device=device)
    tp = torch.tensor([cond['type']], device=device)
    rt = torch.tensor([cond['root']], device=device)
    pb = torch.tensor([cond['pos_bin']], device=device)
    tb = torch.tensor([cond['toend_bin']], device=device)
    phb = torch.tensor([cond['phrase_bin']], device=device)
    cad = torch.tensor([cond['cadence']], device=device)

    best, best_score = None, -1e9
    for c in range(args.candidates):
        gp, grh = model.generate(fn, tp, rt, steps=16, temp=0.85, remask_steps=6,
                                 remask_ratio=0.3, repeat_damp=0.05, expand=False,
                                 pos_bin=pb, toend_bin=tb, phrase_bin=phb, cadence=cad,
                                 seed=rng.randint(0, 10**6))
        pcs = [int(x) for x in gp[0].tolist()]
        rhs = [int(x) for x in grh[0].tolist()]
        # 候选评分: 终止式到达点的和弦音吻合 + 级进率
        score, hit, tot = 0.0, 0, 0
        for i in range(n):
            if cond['phrase_bin'][i] == 0:
                tot += 1
                ivs = CHORD_INTERVALS.get(ID2TYPE.get(cond['type'][i], 'M'), [0, 4, 7])
                if pcs[i] < REST and (pcs[i] - cond['root'][i]) % 12 in ivs:
                    hit += 1
        steps = sum(1 for a, b in zip(pcs, pcs[1:])
                    if a < REST and b < REST and 1 <= min(abs(b - a), 12 - abs(b - a)) <= 2)
        valid = sum(1 for p in pcs if p < REST)
        score = (hit / max(tot, 1)) * 2 + steps / max(valid, 1)
        # 调式一致性: 小调框架惩罚大三度(E), 大调框架惩罚小三度(Eb)与降六级(Ab)
        frame_minor = ID2TYPE.get(cond['type'][0], 'M') in ('m', 'm7', 'dim', 'dim7', 'hdim7')
        if frame_minor:
            bad = sum(1 for p in pcs[:n] if p == 4)
            score -= bad / max(valid, 1) * 1.5
        else:
            bad = sum(1 for p in pcs[:n] if p in (3, 8))
            score -= bad / max(valid, 1) * 1.5
            good3 = sum(1 for p in pcs[:n] if p == 4)
            score += good3 / max(valid, 1) * 0.5
        if score > best_score:
            best_score, best = score, (pcs, rhs)
    pcs, rhs = best
    print(f'生成完成 (候选评分 {best_score:.2f})')

    # 渲染
    arrival_pc = {}   # 到达点的目标和弦根音 (pc)
    for i in range(n):
        if cond['phrase_bin'][i] == 0:
            arrival_pc[i] = cond['root'][i]
    target_pc = [arrival_pc.get(i, cond['root'][i]) for i in range(n)]
    midi = assign_octaves(pcs)
    midi = resolve_cadence_endings(midi, cond, target_pc)

    chords_span = [{'phrase_bin': cond['phrase_bin'][i], 'type': cond['type'][i],
                    'root': cond['root'][i]} for i in range(n)]
    write_midi(chords_span, midi, rhs, out, bpm=args.bpm,
               breath=args.breath, render=args.render)


if __name__ == '__main__':
    main()
