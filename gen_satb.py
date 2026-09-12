"""
四声部生成器（v10 成品入口）—— 给一首众赞歌的 Soprano 配出完整四声部。

与 `train_v10_satb.py --export-only` 的区别：后者只导出 64 切片的验证窗口（评估用），
本脚本能处理**整首作品**：按滑窗逐段生成，每段只取"窗口中部"的预测再拼接，
避免窗口接缝处的语气断裂（与 v9 管线同一思路：滑窗 + 取中部）。

用法
----
    # 用数据集里的某一首（按 id 或序号）
    python gen_satb.py --piece chorale_5 --ckpt v10_aug_vl20.pt

    # 批量 5 首，导出到 data/generated/
    python gen_satb.py --n 5 --seed 3 --ckpt v10_aug_vl20.pt

    # 只要真值版（对照听） / 只要生成版
    python gen_satb.py --piece chorale_5 --mode ref
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
from mido import MetaMessage, Message, MidiFile, MidiTrack
import mido

from model.satb_diffusion import SATBDiffusion, SatbVocab
from train_v10_satb import (COND_KEYS, DATA, HOLD, N_VOICES, RHYTHM_VALUES, TPB,
                            VOICE_NAMES, dur_to_bin, group_split, load_corpus,
                            piece_to_window)

# 合唱音色（General MIDI：52 = Choir Aahs）；也可用 19(教堂管风琴)
PROGRAMS = (52, 52, 52, 52)


def write_midi(path, pitch, rhythm, offs, vocab, bpm=76, programs=PROGRAMS):
    """[V,T] 词表索引 + 切片时间轴 → 四轨 MIDI。"""
    mid = MidiFile(ticks_per_beat=TPB)
    meta = MidiTrack()
    meta.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    meta.append(MetaMessage('time_signature', numerator=4, denominator=4))
    mid.tracks.append(meta)
    V, T = pitch.shape
    for v in range(V):
        tr = MidiTrack()
        tr.append(Message('program_change', program=programs[v], channel=v, time=0))
        notes = []
        for t in range(T):
            p, r = int(pitch[v, t]), int(rhythm[v, t])
            if p >= vocab.n_pitch or r >= HOLD:
                continue
            notes.append([float(offs[t]), p + vocab.lo, RHYTHM_VALUES[min(r, 7)]])
        for i, n in enumerate(notes):
            if i + 1 < len(notes):
                n[2] = min(n[2], max(notes[i + 1][0] - n[0], 0.125))
        ev = []
        for st, midi_p, dur in notes:
            on = int(round(st * TPB))
            ev.append((on, 1, midi_p))
            ev.append((on + max(int(round(dur * TPB)), 1), 0, midi_p))
        ev.sort(key=lambda x: (x[0], x[1]))
        last = 0
        for tick, is_on, mp in ev:
            tr.append(Message('note_on' if is_on else 'note_off', note=mp,
                              velocity=68 if is_on else 0, channel=v,
                              time=max(0, tick - last)))
            last = tick
        tr.append(MetaMessage('end_of_track'))
        mid.tracks.append(tr)
    mid.save(str(path))


@torch.no_grad()
def harmonize_piece(model, piece, vocab, device, win=64, stride=24, keep_soprano=True):
    """整首作品：滑窗生成 + 只取窗口中部，拼成完整四声部。

    返回 (pitch_midi[4,T], rhythm[4,T], offs[T])，offs 为绝对拍位。
    """
    T = len(piece['slices'])
    pitch = torch.full((N_VOICES, T), vocab.rest, dtype=torch.long)
    rhythm = torch.full((N_VOICES, T), HOLD, dtype=torch.long)
    filled = torch.zeros(N_VOICES, T, dtype=torch.bool)
    # 真值（用于已知声部与对照）
    for t, s in enumerate(piece['slices']):
        for v, name in enumerate(VOICE_NAMES):
            m = s['midi'].get(name)
            if m is not None:
                pitch[v, t] = m - vocab.lo
                # 时值必须是真实 bin（与训练口径 piece_to_window 一致）：
                # 写死成某一档会让"已知声部"的节奏与训练不符，真值 MIDI 的长音
                # 也会被截断。
                rhythm[v, t] = dur_to_bin(s['dur']) if s['attack'].get(name) else HOLD
    starts = list(range(0, max(1, T - win + 1), stride))
    if not starts or starts[-1] + win < T:
        starts.append(max(0, T - win))
    for si, ws in enumerate(starts):
        we = min(ws + win, T)
        w = piece_to_window(piece, vocab, ws, we, 0)
        if w is None:
            continue
        p = w['pitch'][None].to(device)
        r = w['rhythm'][None].to(device)
        known = torch.zeros_like(p, dtype=torch.bool)
        if keep_soprano:
            known[:, 0] = True
        cond = {k: w[k][None].to(device) for k in COND_KEYS}
        gp, gr = model.generate(p, r, known, cond, steps=1, argmax=True)
        # 只取窗口中部（首尾各让出 stride/2），避免接缝
        a = 0 if si == 0 else stride // 2
        b = (we - ws) if si == len(starts) - 1 else (we - ws) - stride // 2
        for t in range(a, b):
            gi = ws + t
            if 0 <= gi < T:
                for v in range(N_VOICES):
                    if keep_soprano and v == 0:
                        continue
                    pitch[v, gi] = gp[0, v, t].cpu()
                    rhythm[v, gi] = gr[0, v, t].cpu()
    offs = [float(s['off']) for s in piece['slices']]
    return pitch, rhythm, offs


def piece_truth(piece, vocab):
    """真值四声部（对照听用）。"""
    T = len(piece['slices'])
    pitch = torch.full((N_VOICES, T), vocab.rest, dtype=torch.long)
    rhythm = torch.full((N_VOICES, T), HOLD, dtype=torch.long)
    for t, s in enumerate(piece['slices']):
        for v, name in enumerate(VOICE_NAMES):
            m = s['midi'].get(name)
            if m is not None:
                pitch[v, t] = m - vocab.lo
                # 时值必须是真实 bin（与训练口径 piece_to_window 一致）：
                # 写死成某一档会让"已知声部"的节奏与训练不符，真值 MIDI 的长音
                # 也会被截断。
                rhythm[v, t] = dur_to_bin(s['dur']) if s['attack'].get(name) else HOLD
    return pitch, rhythm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='v10_aug_vl20.pt')
    ap.add_argument('--piece', type=str, default=None, help='如 chorale_5（默认随机抽）')
    ap.add_argument('--n', type=int, default=1, help='批量生成几首')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=480)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--win', type=int, default=64)
    ap.add_argument('--stride', type=int, default=24)
    ap.add_argument('--bpm', type=int, default=76)
    ap.add_argument('--mode', choices=['gen', 'ref', 'both'], default='both')
    ap.add_argument('--heldout', action='store_true',
                    help='只从**留出曲**里抽（模型没训过, 试听更有说服力）')
    ap.add_argument('--out-dir', type=str, default='data/generated')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab = SatbVocab()
    pieces = json.load(open(DATA / 'chorales_satb_v2.json', encoding='utf-8'))
    by_id = {p['id']: p for p in pieces}
    rng = random.Random(args.seed)
    if args.piece:
        todo = [by_id[args.piece]] if args.piece in by_id else []
        if not todo:
            print(f'找不到 {args.piece}（可用 id 如 chorale_0 … chorale_{len(pieces)-1}）')
            return
    else:
        pool = [p for p in pieces if len(p['slices']) >= args.win]
        if args.heldout:
            _, vl, _ = group_split(load_corpus('chorales', vocab, args.win, 8), 0.08)
            ids = {w['piece'] for w in vl}
            pool = [p for p in pool if p['id'] in ids]
            print(f'留出曲 {len(pool)} 首可用')
        todo = rng.sample(pool, min(args.n, len(pool)))

    model = SATBDiffusion(d=args.d, h=8, layers=args.layers, vocab=vocab).to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device, weights_only=True))
    model.eval()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in todo:
        pitch, rhythm, offs = harmonize_piece(model, p, vocab, device,
                                              win=args.win, stride=args.stride)
        base = f"satb_{p['id']}"
        made = []
        if args.mode in ('gen', 'both'):
            write_midi(out_dir / f'{base}_gen.mid', pitch, rhythm, offs, vocab, args.bpm)
            made.append(f'{base}_gen.mid')
        if args.mode in ('ref', 'both'):
            rp, rr = piece_truth(p, vocab)
            write_midi(out_dir / f'{base}_ref.mid', rp, rr, offs, vocab, args.bpm)
            made.append(f'{base}_ref.mid')
        print(f"  {p['id']} ({p['title'][:34]}) {len(p['slices'])} 切片 → {' / '.join(made)}")
    print(f'→ {out_dir}')


if __name__ == '__main__':
    main()
