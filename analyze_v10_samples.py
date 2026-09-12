"""对比 v10 生成版与真值版（同为"给 Soprano 配四声部"）的四声部统计与音乐性。"""
import sys
from pathlib import Path

from mido import MidiFile

ROOT = Path(__file__).parent
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def load(path):
    m = MidiFile(str(path))
    voices = []
    for tr in m.tracks[1:]:
        t = 0
        on = {}
        out = []
        for msg in tr:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                on.setdefault(msg.note, []).append(t)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if on.get(msg.note):
                    st = on[msg.note].pop(0)
                    out.append((st, t, msg.note))
        out.sort()
        voices.append(out)
    return voices


def vl_metrics(voices):
    """相邻声部对: 平行五/八度、交越、间距 > 八度。以切片对齐（同时刻的音）。"""
    par5 = par8 = cross = space = pairs = 0
    n = min(len(v) for v in voices) if voices else 0
    for v in range(len(voices) - 1):
        hi, lo = voices[v], voices[v + 1]
        k = min(len(hi), len(lo))
        for i in range(k - 1):
            a0, a1 = hi[i][2], hi[i + 1][2]
            c0, c1 = lo[i][2], lo[i + 1][2]
            pairs += 1
            if c0 > a0 or c1 > a1:
                cross += 1
            if abs(a0 - c0) > 12 and v < 2:
                space += 1
            da, dc = a1 - a0, c1 - c0
            if da and dc and (da > 0) == (dc > 0):
                if (a0 - c0) % 12 == 7 and (a1 - c1) % 12 == 7:
                    par5 += 1
                elif (a0 - c0) % 12 == 0 and (a1 - c1) % 12 == 0:
                    par8 += 1
    return dict(pairs=pairs, par5=par5, par8=par8, cross=cross, space=space)


def describe(tag, voices):
    note_counts = [len(v) for v in voices]
    rngs = [f'{NAMES[min(p for _, _, p in v) % 12]}{min(p for _, _, p in v) // 12 - 1}'
            f'-{NAMES[max(p for _, _, p in v) % 12]}{max(p for _, _, p in v) // 12 - 1}'
            for v in voices if v]
    m = vl_metrics(voices)
    print(f'  {tag}: {len(voices)} 声部 | 各声部音符 {note_counts} | 音域 {rngs}')
    print(f'       声部进行: 平行五 {m["par5"]} / 平行八度 {m["par8"]} / 交越 {m["cross"]}'
          f' / 间距>八度 {m["space"]}  (共 {m["pairs"]} 对)')


import glob

print('=== 整首四声部成品 (gen_satb.py 导出) ===')
for g in sorted(glob.glob(str(ROOT / 'data/generated/satb_*_gen.mid'))):
    r = g.replace('_gen.mid', '_ref.mid')
    name = Path(g).stem.replace('satb_', '').replace('_gen', '')
    print(f'--- {name} ---')
    describe('生成', load(Path(g)))
    if Path(r).exists():
        describe('真值', load(Path(r)))

print()
print('=== 评估窗口样例 (v10_gen_*/v10_ref_*) ===')
for k in (1, 2, 3):
    if (ROOT / f'data/generated/v10_gen_{k}.mid').exists():
        print(f'--- 第 {k} 例 ---')
        describe('生成', load(ROOT / f'data/generated/v10_gen_{k}.mid'))
        describe('真值', load(ROOT / f'data/generated/v10_ref_{k}.mid'))
