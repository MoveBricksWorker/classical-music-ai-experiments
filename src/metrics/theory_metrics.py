"""
乐理合格度指标 —— 参考 MusicAIR (#16) 三件套，用 mido + numpy 实现。

    key_confidence    : Krumhansl-Schmuckler 调性判定 (MusicAIR: AI 0.85 vs 人类 0.79)
    melodic_smoothness: 平均音程 / step ratio (>0.6) / 方向变化率 / 大跳率
    rhythm_matching   : 强拍对齐度 (强拍起音率 / 长音强拍起始率 / 强拍空拍率)

输入支持两种形式:
    1. MIDI 文件路径 (解析指定 track/channel)
    2. token 流 (pitch_class 列表 + rhythm_id 列表, 语义与 src/constants.py 一致:
       pitch_class 0-11 音高, >=12 休止; rhythm_id 见 MELODY_RHYTHM)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from constants import MELODY_RHYTHM  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 基础数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NoteEvent:
    start_beat: float
    pitch: int          # MIDI pitch (60 = C5)
    dur_beats: float


REST_PC = 12            # token 流中 >= 该值视为休止

# Krumhansl-Kessler 调性剖面 (Krumhansl & Kessler 1982)
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _kk_profiles() -> dict[str, np.ndarray]:
    """24 个调 (12 大调 + 12 小调) 的 K-K 剖面。"""
    out = {}
    for tonic in range(12):
        out[f'{_NAMES[tonic]} major'] = np.roll(_KK_MAJOR, tonic)
        out[f'{_NAMES[tonic]} minor'] = np.roll(_KK_MINOR, tonic)
    return out


KK_PROFILES = _kk_profiles()


# ─────────────────────────────────────────────────────────────────────────────
# 输入解析
# ─────────────────────────────────────────────────────────────────────────────

def load_midi_notes(midi_path: str, track: int = 0) -> list[NoteEvent]:
    """从 MIDI 文件解析指定音轨的音符为 (start_beat, pitch, dur_beats)。"""
    import mido
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat or 480

    events = []  # (abs_tick, 'on'/'off', pitch)
    abs_tick = 0
    for msg in mid.tracks[track]:
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            events.append((abs_tick, 'on', msg.note))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            events.append((abs_tick, 'off', msg.note))

    # 配对 on/off (tick → beat)
    open_notes: dict[int, float] = {}
    notes: list[NoteEvent] = []
    for tick, etype, pitch in events:
        if etype == 'on':
            open_notes[pitch] = tick
        else:
            start = open_notes.pop(pitch, None)
            if start is None:
                continue
            start_beat = start / tpb
            dur_beats = (tick - start) / tpb
            if dur_beats > 0:
                notes.append(NoteEvent(start_beat, pitch, dur_beats))
    notes.sort(key=lambda n: n.start_beat)
    return notes


def tokens_to_notes(pitches: list[int], rhythms: list[int]) -> list[NoteEvent]:
    """token 流 → NoteEvent 列表。pc>=12 为休止; rhythm_id → 时值 (beat)。"""
    notes = []
    t = 0.0
    for pc, rh in zip(pitches, rhythms):
        dur = MELODY_RHYTHM.get(rh, 0.5)
        if pc < REST_PC:
            notes.append(NoteEvent(t, pc % 12 + 60, dur))  # 归一化到 C5 八度
        t += dur
    return notes


def _pc_hist_duration_weighted(notes: list[NoteEvent]) -> np.ndarray:
    """时值加权的 12 维音级直方图 (归一化)。"""
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += n.dur_beats
    s = hist.sum()
    return hist / s if s > 0 else hist


# ─────────────────────────────────────────────────────────────────────────────
# 1. Key confidence (Krumhansl-Schmuckler)
# ─────────────────────────────────────────────────────────────────────────────

def key_confidence(pc_hist: np.ndarray) -> dict:
    """调性判定 + 置信度。

    对 24 个 K-K 剖面做 Pearson 相关, 取最大者为调性,
    confidence = 最大相关系数 (与 MusicAIR 的口径一致)。
    额外返回 top-1 与 top-2 的差距 (ambiguity margin)。
    """
    scores = {k: float(np.corrcoef(pc_hist, v)[0, 1]) for k, v in KK_PROFILES.items()}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    key, conf = ranked[0]
    return {
        'key': key,
        'confidence': conf,
        'second_key': ranked[1][0],
        'second_conf': ranked[1][1],
        'ambiguity_margin': ranked[0][1] - ranked[1][1],
    }


def evaluate_key(notes: list[NoteEvent]) -> dict:
    return key_confidence(_pc_hist_duration_weighted(notes))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Melodic smoothness
# ─────────────────────────────────────────────────────────────────────────────

def melodic_smoothness(notes: list[NoteEvent]) -> dict:
    """平均音程 / step ratio / 方向变化率 / 大跳率。

    休止处断开音程累计 (不跨休止计算音程)。
    音级空间 (token 输入) 最大距离为 6 半音 (三全音), 故大跳阈值取 >=5
    (四度及以上); MIDI 输入保留八度信息, 该阈值仍可比。
    """
    intervals = []
    prev = None
    for n in notes:
        pc = n.pitch % 12
        if prev is not None:
            d = abs(pc - prev)
            d = min(d, 12 - d)
            intervals.append(d)
        prev = pc

    if not intervals:
        return {'avg_interval': 0.0, 'step_ratio': 0.0,
                'direction_change_rate': 0.0, 'leap_ratio': 0.0, 'n_intervals': 0}

    arr = np.array(intervals, dtype=float)
    step_ratio = float((arr <= 2).mean())
    leap_ratio = float((arr >= 5).mean())

    # 方向变化率: 用带符号的位移序列 (在 12 音级环上取最短路径)
    signed = []
    prev = None
    for n in notes:
        pc = n.pitch % 12
        if prev is not None:
            raw = pc - prev
            if raw > 6:
                raw -= 12
            elif raw < -6:
                raw += 12
            signed.append(raw)
        prev = pc
    dir_changes = sum(
        1 for a, b in zip(signed[:-1], signed[1:]) if a != 0 and b != 0 and a * b < 0
    )
    nonzero_pairs = sum(1 for a, b in zip(signed[:-1], signed[1:]) if a != 0 and b != 0)

    return {
        'avg_interval': float(arr.mean()),
        'step_ratio': step_ratio,
        'direction_change_rate': dir_changes / nonzero_pairs if nonzero_pairs else 0.0,
        'leap_ratio': leap_ratio,
        'n_intervals': len(intervals),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Rhythm matching (强拍-重音对齐)
# ─────────────────────────────────────────────────────────────────────────────

def rhythm_matching(notes: list[NoteEvent], beats_per_measure: int = 4,
                    beat_shift: float = 0.0) -> dict:
    """强拍对齐度三指标。

    - strong_onset_ratio   : 起音落在强拍 (每小节第 1、3 拍) 的比例
    - long_on_strong_ratio : 长音 (>=1 拍) 起始于强拍的比例
    - strong_beat_gap_ratio: 强拍位置无任何起音的空拍比例 (越低越对齐)
    - downbeat_onset_ratio : 起音落在每小节第 1 拍的比例

    beat_shift: 网格相位偏移 (拍)。拍位条件模型的评估需与条件同一拍位框架,
    传入 -首音真实拍位 mod 4; 无拍位条件的模型用 0 (从 0 起拍)。
    """
    m = beats_per_measure
    shift = int(round(beat_shift * 4))
    strong_mods = {(shift + 0) % (m * 4), (shift + 2 * 4) % (m * 4)}
    down_mod = (shift + 0) % (m * 4)

    onsets = [n.start_beat for n in notes]
    strong_onset = sum(1 for b in onsets if round(b * 4) % (m * 4) in strong_mods)
    downbeat_onset = sum(1 for b in onsets if round(b * 4) % (m * 4) == down_mod)
    long_notes = [n for n in notes if n.dur_beats >= 1.0]
    long_on_strong = sum(1 for n in long_notes
                         if round(n.start_beat * 4) % (m * 4) in strong_mods)

    # 强拍空拍率: 统计覆盖范围内强拍位置 (移位后的框架, 四分音符单位)
    if onsets:
        onset_set = {round(b * 4) for b in onsets}
        max_q = max(onset_set)
        strong_qs = [q for q in range(0, max_q + 1) if q % (m * 4) in strong_mods]
        strong_hit = sum(1 for q in strong_qs if q in onset_set)
        gap_ratio = 1.0 - strong_hit / len(strong_qs) if strong_qs else 0.0
    else:
        gap_ratio = 1.0

    n_on = len(onsets)
    return {
        'strong_onset_ratio': strong_onset / n_on if n_on else 0.0,
        'downbeat_onset_ratio': downbeat_onset / n_on if n_on else 0.0,
        'long_on_strong_ratio': long_on_strong / len(long_notes) if long_notes else 0.0,
        'strong_beat_gap_ratio': gap_ratio,
        'n_onsets': n_on,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 汇总入口
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_notes(notes: list[NoteEvent]) -> dict:
    return {
        'key': evaluate_key(notes),
        'smoothness': melodic_smoothness(notes),
        'rhythm': rhythm_matching(notes),
        'n_notes': len(notes),
    }


def evaluate_midi(midi_path: str, track: int = 0) -> dict:
    """对 MIDI 文件完整评估 (调性/平滑度/节奏)。"""
    notes = load_midi_notes(midi_path, track=track)
    return evaluate_notes(notes)


def evaluate_tokens(pitches: list[int], rhythms: list[int]) -> dict:
    """对 token 流完整评估。"""
    return evaluate_notes(tokens_to_notes(pitches, rhythms))


def evaluate_pieces(pieces: list, beats_per_measure: int = 4) -> dict:
    """对多首乐曲的 token 流逐曲评估并聚合 (均值 ± 标准差)。

    pieces: [(pitches, rhythms), ...] 或 [(pitches, rhythms, beats), ...]
    beats 为小节内拍位序列时, 节奏指标按该框架对齐
    (shift = -首音拍位 mod 4), 与拍位条件模型的输入口径一致。
    """
    key_confs, key_keys = [], []
    avg_ints, step_ratios, dir_rates, leap_ratios = [], [], [], []
    s_onset, d_onset, l_strong, gaps = [], [], [], []
    for item in pieces:
        pitches, rhythms = item[0], item[1]
        beat_shift = 0.0
        if len(item) > 2 and item[2] and item[2][0] is not None:
            beat_shift = (-item[2][0]) % beats_per_measure
        notes = tokens_to_notes(pitches, rhythms)
        if not notes:
            continue
        k = evaluate_key(notes)
        key_confs.append(k['confidence'])
        key_keys.append(k['key'])
        sm = melodic_smoothness(notes)
        avg_ints.append(sm['avg_interval'])
        step_ratios.append(sm['step_ratio'])
        dir_rates.append(sm['direction_change_rate'])
        leap_ratios.append(sm['leap_ratio'])
        rm = rhythm_matching(notes, beats_per_measure, beat_shift)
        s_onset.append(rm['strong_onset_ratio'])
        d_onset.append(rm['downbeat_onset_ratio'])
        l_strong.append(rm['long_on_strong_ratio'])
        gaps.append(rm['strong_beat_gap_ratio'])

    def agg(x):
        x = np.array(x)
        return {'mean': float(x.mean()), 'std': float(x.std()), 'n': len(x)}

    from collections import Counter
    return {
        'n_pieces': len(key_confs),
        'key': {**agg(key_confs), 'mode': Counter(key_keys).most_common(1)[0][0]},
        'smoothness': {
            'avg_interval': agg(avg_ints),
            'step_ratio': agg(step_ratios),
            'direction_change_rate': agg(dir_rates),
            'leap_ratio': agg(leap_ratios),
        },
        'rhythm': {
            'strong_onset_ratio': agg(s_onset),
            'downbeat_onset_ratio': agg(d_onset),
            'long_on_strong_ratio': agg(l_strong),
            'strong_beat_gap_ratio': agg(gaps),
        },
    }


if __name__ == '__main__':
    import json
    d = json.load(open('data/processed/melody_llm_full_v2.json', encoding='utf-8'))
    # 按 prev_pc=null 边界切分为乐曲
    pieces, cur_p, cur_r = [], [], []
    for n in d:
        if n.get('prev_pc') is None and cur_p:
            pieces.append((cur_p, cur_r)); cur_p, cur_r = [], []
        cur_p.append(n['pc']); cur_r.append(n['rhythm'])
    if cur_p:
        pieces.append((cur_p, cur_r))
    res = evaluate_pieces(pieces)
    print(json.dumps(res, ensure_ascii=False, indent=2))
