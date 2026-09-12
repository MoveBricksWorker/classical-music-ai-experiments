"""MIDI → WAV/MP3 渲染（本机无软音源/音色库时的替代方案）。

用加法合成（谐波叠加 + 起落包络 + 轻混响）把四声部 MIDI 渲成可播放音频，
目的是**盲听对照**（生成 vs 真巴赫），不是高保真音源。

    python render_audio.py data/generated/full/full_1.mid
    python render_audio.py data/generated/full/*.mid data/generated/satb_chorale_6_ref.mid
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

SR = 44100
HARMONICS = np.array([1.0, 0.50, 0.28, 0.15, 0.085, 0.05, 0.03])
GAIN_BY_CHANNEL = [1.0, 0.80, 0.82, 0.95]     # S / A / T / B
PAN_BY_CHANNEL = [0.12, -0.10, 0.10, -0.12]


def read_notes(path: Path):
    """→ (notes, total_seconds)；notes = [(start, dur, pitch, channel)]，时间单位秒。"""
    mid = mido.MidiFile(str(path))
    tempo = 500000
    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break
    sec_per_tick = tempo / 1e6 / mid.ticks_per_beat
    notes, end_tick = [], 0
    for tr in mid.tracks:
        tick, pending = 0, {}
        for msg in tr:
            tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                pending.setdefault(msg.note, []).append(tick)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                q = pending.get(msg.note)
                if q:
                    st = q.pop(0)
                    notes.append((st * sec_per_tick, (tick - st) * sec_per_tick,
                                  msg.note, msg.channel))
                    end_tick = max(end_tick, tick)
    return notes, end_tick * sec_per_tick


def synth(notes, total_sec: float):
    n = int((total_sec + 1.6) * SR)
    buf = np.zeros((n, 2))
    for st, dur, pitch, ch in notes:
        i0, i1 = int(st * SR), min(n, int((st + max(dur, 0.06)) * SR))
        if i1 <= i0:
            continue
        t = np.arange(i1 - i0) / SR
        f = 440.0 * 2 ** ((pitch - 69) / 12)
        sig = np.zeros_like(t)
        for k, a in enumerate(HARMONICS, start=1):
            sig += a * np.sin(2 * np.pi * f * k * t)
        sig += 0.35 * np.sin(2 * np.pi * f * 1.002 * t)     # 轻微失谐加厚
        env = np.exp(-0.30 * t)
        na = min(int(0.018 * SR), len(t))
        env[:na] *= np.linspace(0, 1, na)
        nr = min(int(0.09 * SR), len(t))
        env[-nr:] *= np.linspace(1, 0, nr)
        sig *= env * (GAIN_BY_CHANNEL[ch % 4] / (HARMONICS.sum() + 0.35))
        pan = PAN_BY_CHANNEL[ch % 4]
        gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        buf[i0:i1, 0] += sig * gl
        buf[i0:i1, 1] += sig * gr
    return buf


def reverb(buf, decay=1.1, mix=0.20, seed=0):
    """简易卷积混响（指数衰减噪声脉冲响应），让四声部不至于太干。"""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(int(decay * SR)) * np.exp(-4.5 * np.arange(int(decay * SR)) / (decay * SR))
    ir[0] = 1.0
    ir /= np.abs(ir).sum() / 12
    wet = np.stack([fftconvolve(buf[:, c], ir)[:len(buf)] for c in range(2)], axis=1)
    return buf * (1 - mix) + wet * mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mids', nargs='+')
    ap.add_argument('--out-dir', default='data/generated/audio')
    ap.add_argument('--mp3', action='store_true', help='同时导出 mp3（需要 ffmpeg）')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in args.mids:
        p = Path(m)
        notes, total = read_notes(p)
        if not notes:
            print(f'跳过（无音符）: {p.name}')
            continue
        buf = reverb(synth(notes, total))
        peak = float(np.abs(buf).max())
        buf = np.tanh(buf / peak * 1.15) * 0.89            # 归一 + 软限幅
        wav = out_dir / f'{p.stem}.wav'
        sf.write(str(wav), buf, SR, subtype='PCM_16')
        msg = f'{p.name} → {wav}  ({len(notes)} 音, {total:.1f}s)'
        if args.mp3 and shutil.which('ffmpeg'):
            mp3 = out_dir / f'{p.stem}.mp3'
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav),
                            '-b:a', '192k', str(mp3)], check=True)
            msg += f' + {mp3.name}'
        print(msg)


if __name__ == '__main__':
    main()
