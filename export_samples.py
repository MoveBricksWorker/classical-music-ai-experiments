"""
把扩散模型生成的旋律样例导出为可听 MIDI (旋律轨 + 和弦垫轨)。

用法:
    python export_samples.py [--json data/generated/diffusion_samples.json]
                             [--out data/generated/]

每首样例: 旋律 (ch0) + 三音和弦垫 (ch1), 120 BPM, 4/4。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

import argparse, json

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from constants import MELODY_RHYTHM, CHORD_INTERVALS, T2ID

TPB = 480
ID2TYPE = {v: k for k, v in T2ID.items()}


def sample_to_midi(sample: dict, out_path: Path):
    pitches = sample['pitch']
    rhythms = sample['rhythm']
    harmony = sample.get('harmony')  # 可选: [{'root':, 'type_id':} x 每和弦]

    mid = MidiFile(ticks_per_beat=TPB)
    t0 = MidiTrack()
    t0.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
    t0.append(MetaMessage('time_signature', numerator=4, denominator=4))
    mid.tracks.append(t0)

    # 旋律轨 (delta 时间)
    mel_events = []
    tick = 0
    for pc, rh in zip(pitches, rhythms):
        dur = MELODY_RHYTHM.get(rh, 0.5)
        ticks = int(dur * TPB)
        if pc < 12:
            mel_events.append((tick, 'on', pc + 72))
            mel_events.append((tick + max(ticks, 1), 'off', pc + 72))
        tick += ticks
    mel_events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

    mel = MidiTrack()
    mel.append(Message('program_change', program=0, channel=0, time=0))
    last = 0
    for abs_t, etype, note in mel_events:
        mel.append(Message('note_on' if etype == 'on' else 'note_off',
                           note=note,
                           velocity=72 if etype == 'on' else 0,
                           channel=0, time=abs_t - last))
        last = abs_t
    mel.append(MetaMessage('end_of_track'))

    # 和弦垫轨: 每 4 音一个和弦, 持续 4 拍
    pad_events = []
    t = 0
    n_chords = len(pitches) // 4
    for ci in range(n_chords):
        if harmony and ci < len(harmony):
            h = harmony[ci]
            r, ct = h.get('root', 0), ID2TYPE.get(h.get('type_id', 0), 'M')
        else:
            r, ct = 0, 'M'
        for iv in sorted(CHORD_INTERVALS.get(ct, [0, 4, 7])):
            pad_events.append((t, 'on', r + 60 + iv))
            pad_events.append((t + 4 * TPB, 'off', r + 60 + iv))
        t += 4 * TPB
    pad_events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

    pad = MidiTrack()
    pad.append(Message('program_change', program=0, channel=1, time=0))
    last = 0
    for abs_t, etype, note in pad_events:
        pad.append(Message('note_on' if etype == 'on' else 'note_off',
                           note=note,
                           velocity=56 if etype == 'on' else 0,
                           channel=1, time=abs_t - last))
        last = abs_t
    pad.append(MetaMessage('end_of_track'))

    mid.tracks.append(mel)
    mid.tracks.append(pad)
    mid.save(str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='data/generated/diffusion_samples.json')
    parser.add_argument('--out', type=str, default='data/generated')
    args = parser.parse_args()

    data = json.load(open(ROOT / args.json, encoding='utf-8'))
    samples = data['samples']
    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)
    for i, s in enumerate(samples):
        sample_to_midi(s, out_dir / f'sample_{i:03d}.mid')
    print(f'导出 {len(samples)} 个 MIDI → {out_dir}')


if __name__ == '__main__':
    main()
