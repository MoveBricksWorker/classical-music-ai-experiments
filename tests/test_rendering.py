"""渲染层回归测试 —— 固定"三轨 MIDI 网格契约"与"句末拉长可听"。

背景: 旧渲染把每个旋律音截断到 ≤1 拍, 模型学到的句末拉长 (rhythm bin 6-7)
进不了 MIDI; 修成"小节内按节奏 bin 比例分配"后需要防止回归。

用法: python tests/test_rendering.py
"""
import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))

from mido import MidiFile
from gen_sentence import write_midi, render_melody_timing
from constants import MELODY_RHYTHM


def _melody_notes(path):
    mid = MidiFile(str(path))
    tr = mid.tracks[1]
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
    return mid, out


def test_timing_shares():
    """小节内按比例分配、吸附八分网格、句末留气口、无重叠。"""
    rhs = [4, 4, 4, 4, 7, 2, 2, 2]        # 第 2 小节首音为全音符 bin
    is_end = [False, False, False, True, False, False, False, True]
    timing = render_melody_timing(rhs, is_end, breath=0.25)
    d = {i: (on, dur) for i, on, dur in timing}
    # 第 1 小节: 四个四分 → 各 1 拍; 第 4 音是句末 → 留 0.25 拍气口
    assert abs(d[0][1] - 1.0) < 1e-9, d[0]
    assert abs(d[3][1] - 0.75) < 1e-9, d[3]
    # 第 2 小节: bin7 长音明显长于同小节 bin2 短音
    assert d[4][1] > d[5][1] + 1.0, (d[4], d[5])
    # 起点全部吸附在十六分网格 (0.25 拍整数倍) —— 不偏格
    assert all(abs(on / 0.25 - round(on / 0.25)) < 1e-9 for _, on, _ in timing)
    # 不重叠
    seq = sorted((on, on + dur) for _, on, dur in timing)
    for (s1, e1), (s2, _) in zip(seq[:-1], seq[1:]):
        assert e1 <= s2 + 1e-9, (s1, e1, s2)
    print('timing shares OK: 句末长音', round(d[4][1], 2), '拍')


def test_midi_contract():
    """三轨齐全 / 网格对齐 / 句末音更长 / 长音真实存在。"""
    n = 16
    chords_span = [{'phrase_bin': 0 if i in (7, 15) else 2, 'type': 0, 'root': 0}
                   for i in range(n)]
    pcs = [60 + (i % 5) for i in range(n)]
    rhs = [4, 4, 4, 4, 4, 4, 4, 7, 2, 2, 2, 2, 2, 2, 2, 7]
    out = ROOT / 'data/generated/_render_test.mid'
    write_midi(chords_span, pcs, rhs, out, bpm=76, breath=0.25, render='rhythm')

    mid, notes = _melody_notes(out)
    assert len(mid.tracks) == 4, f'应为 4 轨 (tempo+3), 实际 {len(mid.tracks)}'
    # 极短音可能被吸附到同格而合并 (不会重叠/零时值)
    assert 0 < len(notes) <= n, len(notes)

    # 无重叠 (音符按起点排序, 前一个的结束不超过后一个的起点)
    for (s1, e1, _), (s2, _, _) in zip(notes[:-1], notes[1:]):
        assert e1 <= s2, (s1, e1, s2)

    # 网格: 所有起点落在八分音符 (TPB/2) 上 —— 三轨不偏格
    for s, _, _ in notes:
        assert s % (mid.ticks_per_beat // 4) == 0, s

    # 小节对齐: 每小节 4 拍
    assert notes[-1][1] <= 16 * mid.ticks_per_beat

    durs = [(e - s) / mid.ticks_per_beat for s, e, _ in notes]
    end_durs = [durs[i] for i in (7, 15)]
    mid_durs = [durs[i] for i in range(n) if i not in (7, 15)]
    assert statistics.mean(end_durs) > statistics.mean(mid_durs), (end_durs, mid_durs)
    assert max(durs) > 1.5, '句末长音应真实进入 MIDI'
    print(f'midi contract OK: 句末均 {statistics.mean(end_durs):.2f} 拍 vs 句中 '
          f'{statistics.mean(mid_durs):.2f} 拍, 最长 {max(durs):.2f} 拍')


def test_slot_mode_still_available():
    """旧口径 (slot) 仍可复现, 用于对照历史成品。"""
    n = 8
    chords_span = [{'phrase_bin': 2, 'type': 0, 'root': 0} for _ in range(n)]
    out = ROOT / 'data/generated/_render_test_slot.mid'
    write_midi(chords_span, [60] * n, [4] * n, out, render='slot')
    _, notes = _melody_notes(out)
    durs = [(e - s) / 480 for s, e, _ in notes]
    assert all(abs(d - 1.0) < 1e-9 for d in durs), durs
    print('slot mode OK (旧口径可复现)')


if __name__ == '__main__':
    test_timing_shares()
    test_midi_contract()
    test_slot_mode_still_available()
    print('渲染测试全部通过 ✔')
