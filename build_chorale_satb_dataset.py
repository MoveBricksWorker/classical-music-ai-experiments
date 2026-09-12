"""
众赞歌 SATB 四声部数据集 + 织体标注 (v2)。

与 v1 (`chorales_sentences_v1.json`) 的关系
------------------------------------------------
v1 只用了 Soprano 一条旋律，其余 3/4 的音乐信息被丢掉。v2 保留**全部四个声部**，
并以**纵向切片**为单位组织：每个 onset = 一个切片，含四声部当时正在发声音的
音高与"是否在此起音"。这是四声部生成/声部补全任务的标准表示
（DeepBach / BachBot 口径），也是本项目从"旋律生成"走向"多声部写作"的数据基础。

    - 调性/移调到 C、和声 (func/type/root)、乐句与终止式：**复用 v1 的算法**
      （`build_chorale_dataset.build_piece`），保证与 v9 的实验可比；
    - 新增：四声部切片、声部起音标记、切片级和声、小节级伴奏音型、
      乐句级织体类型与支撑特征。

织体标注（本次新增的重点）
------------------------------------------------
三个层级，全部**规则可复现**（Ollama 只用于事后审计，见 `annotate_texture.py`）：

1. 切片级：`n_voices`（同时发声音数）、`attack`（谁在此起音）、
   `unison`（有同音级重合）、`doubling`（八度/同度加倍）；
2. 小节级 `figure`（伴奏音型，为后续钢琴语料预留）：
   `alberti`（根-五-三-五循环）/ `broken`（低高交替）/ `arpeggio`（同向和弦音级进）/
   `sustained`（长音保持）/ `block`（柱式，≥3 声部同起音）/ `moving` / `none`；
3. 乐句级 `texture`：
   `pedal`（持续音主导）/ `imitative`（声部间有动机模仿）/
   `chordal`（柱式和弦，同起音率高）/ `unison_octave`（齐奏/八度为主）/
   `polyphonic`（节奏独立、无模仿）/ `mixed`，
   并附特征：同起音率、节奏独立性、模仿得分、持续音占比、齐奏率、平均声部数。

输出
------------------------------------------------
    data/processed/chorales_satb_v2.json   数据集本体
    data/processed/chorales_satb_v2_stats.json  统计摘要

用法
------------------------------------------------
    python build_chorale_satb_dataset.py [--limit N] [--debug]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from music21 import chord as m21chord, corpus, meter as m21meter, note as m21note, roman, stream

from build_chorale_dataset import DEGREE_FUNC, build_piece, rn_to_type
from constants import CHORD_INTERVALS

VOICES = ['S', 'A', 'T', 'B']
CANON = {'soprano': 'S', 'alto': 'A', 'tenor': 'T', 'bass': 'B'}
ALIAS = {'cantus': 'S', 'discantus': 'S', 'superius': 'S', 'altus': 'A',
         'contratenor': 'A', 'bassus': 'B', 'basso': 'B', 'basse': 'B'}


# ─────────────────────────────────────────────────────────────────────────────
# 声部提取
# ─────────────────────────────────────────────────────────────────────────────

def _match_voice(part) -> str | None:
    """partName → 'S'/'A'/'T'/'B'；无法判定返回 None。"""
    name = (part.partName or part.id or '')
    low = name.lower()
    for key, v in CANON.items():
        if key in low:
            return v
    for key, v in ALIAS.items():
        if key in low:
            return v
    return None


def _mean_pitch(part) -> float:
    ps = [float(x.pitch.midi) for x in part.flat.notes
          if isinstance(x, m21note.Note) or (isinstance(x, m21chord.Chord) and x.pitches)]
    return sum(ps) / len(ps) if ps else 0.0


def extract_voices(ch) -> dict[str, stream.Part] | None:
    """取出 SATB 四个声部（丢弃乐器加倍声部/数字低音）。

    名字匹配失败时按**平均音高排序**兜底（低→高 = B/T/A/S）：文艺复兴与
    巴洛克语料里常见"两个 Tenor"或声部名不规范的情况（如 Palestrina 的
    Credo 段），只靠名字会丢掉约 1/5 的可用数据。
    """
    cand: dict[str, list] = {v: [] for v in VOICES}
    for p in ch.parts:
        v = _match_voice(p)
        if v:
            cand[v].append(p)
    # 同名多声部（如 'Soprano Oboe 1 Violin1'）取音符最多的那条
    out = {}
    for v in VOICES:
        if not cand[v]:
            break
        best = max(cand[v], key=lambda p: len(list(p.flat.notes)))
        out[v] = best
    if len(out) == 4:
        return out
    # 兜底: 取音符最多的 4 条声部, 按平均音高排序 (最低音→B, 最高音→S)
    parts = [p for p in ch.parts if len(list(p.flat.notes)) > 0]
    if len(parts) >= 4:
        top = sorted(parts, key=lambda p: len(list(p.flat.notes)), reverse=True)[:4]
        order = sorted(top, key=_mean_pitch)
        return dict(zip(['B', 'T', 'A', 'S'], order))
    # 再兜底: spine_N 命名（music21 对无名声部的编号），按 N 排序 = 高→低
    spines = [(p.partName or '') for p in ch.parts]
    if all(s.startswith('spine_') for s in spines) and len(spines) == 4:
        order = sorted(ch.parts, key=lambda p: int((p.partName or 'spine_0').split('_')[1]))
        return dict(zip(VOICES, order))
    return None


def part_events(part) -> list[tuple[float, float, int]]:
    """声部 → [(start, end, midi)]，和弦取与前音最接近的音（声部进行连续）。"""
    ev = []
    prev = None
    for n in part.flat.notes:
        if isinstance(n, m21note.Note):
            p = int(n.pitch.midi)
        elif isinstance(n, m21chord.Chord):
            ps = [int(x.midi) for x in n.pitches]
            p = min(ps, key=lambda m: abs(m - prev)) if prev is not None else ps[0]
        else:
            continue
        st = float(n.getOffsetInHierarchy(part))
        ev.append((st, st + float(n.quarterLength), p))
        prev = p
    ev.sort()
    return ev


# ─────────────────────────────────────────────────────────────────────────────
# 切片构建
# ─────────────────────────────────────────────────────────────────────────────

def build_slices(events: dict[str, list]) -> list[dict]:
    """四声部事件 → 纵向切片序列。

    每个切片记录: off / dur / 各声部当前 MIDI(None=休止) / 是否在该切片起音。
    """
    onsets = sorted({round(e[0], 3) for ev in events.values() for e in ev})
    slices = []
    for i, off in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else None
        midi, attack = {}, {}
        for v in VOICES:
            cur, att = None, False
            for (st, en, p) in events[v]:
                if abs(st - off) < 1e-6:
                    cur, att = p, True
                    break
                if st < off - 1e-6 and en > off + 1e-6:
                    cur = p
                    break
                if st > off:
                    break
            if cur is None:                      # 用最后一个已结束音的前音判断延音
                for (st, en, p) in events[v]:
                    if abs(en - off) < 1e-6:
                        cur = p
                        break
            midi[v], attack[v] = cur, att
        if nxt is None:
            nxt = max((e[1] for ev in events.values() for e in ev), default=off)
        slices.append({'off': round(off, 3), 'dur': round(nxt - off, 3),
                       'midi': {v: midi[v] for v in VOICES},
                       'attack': {v: attack[v] for v in VOICES}})
    return slices


def annotate_slices(slices: list[dict], harm_events: list[tuple]) -> None:
    """补上 n_voices / unison / doubling / 和声 (原地修改)。"""
    def harm_at(off):
        cur = harm_events[0] if harm_events else None
        for h in harm_events:
            if h[0] <= off + 1e-6:
                cur = h
            else:
                break
        return cur

    for s in slices:
        ps = [m for m in s['midi'].values() if m is not None]
        s['n_voices'] = len(ps)
        pcs = [p % 12 for p in ps]
        # pc_collision: 有同音级重合（含八度加倍，四声部写作常态）
        # uniform_pc:   整片只有一个音级（齐奏/八度齐奏，真正的"单声部织体"）
        s['pc_collision'] = len(pcs) != len(set(pcs))
        s['uniform_pc'] = len(set(pcs)) == 1 and len(ps) >= 2
        s['doubling'] = any(abs(a - b) % 12 == 0 and abs(a - b) >= 12
                            for i, a in enumerate(ps) for b in ps[i + 1:])
        h = harm_at(s['off'])
        if h:
            _, func, ctype, root, deg = h
            s['func'], s['type'], s['root'], s['degree'] = func, ctype, root, deg
        else:
            s['func'] = s['type'] = s['root'] = s['degree'] = None


# ─────────────────────────────────────────────────────────────────────────────
# 织体特征与标注
# ─────────────────────────────────────────────────────────────────────────────

def _interval_seq(midis: list[int | None]) -> list[tuple[float, int]]:
    """[(起音时刻, 音程)]，跳过失音。"""
    out = []
    prev = None
    for off, m in midis:
        if m is None:
            continue
        if prev is not None:
            out.append((off, m - prev))
        prev = m
    return out


def imitation_score(events: dict[str, list], spans: list[tuple[float, float]],
                    min_run: int = 4) -> tuple[bool, int]:
    """声部间动机模仿检测。

    判定要求（避免把"两个声部碰巧同向级进"误判成模仿）：
      - 匹配的是 (音程, 时值) 串，即节奏型也要一致；
      - 两声部之间有明显时差 (0.5–4 拍, 覆盖文艺复兴复调的进入间隔) 且时差一致 (±0.5 拍)；
      - 连续匹配 ≥ `min_run` 个音。
    """
    best = 0
    for t0, t1 in spans:
        seqs = {}
        for v in VOICES:
            notes = [(st, en - st, p) for (st, en, p) in events[v] if t0 - 1e-6 <= st < t1]
            out = []
            for k in range(1, len(notes)):
                if notes[k][1] > 0:
                    out.append((notes[k][0], notes[k][2] - notes[k - 1][2], notes[k][1]))
            seqs[v] = out
        for i, va in enumerate(VOICES):
            for vb in VOICES[i + 1:]:
                A, B = seqs[va], seqs[vb]
                for ia in range(len(A)):
                    for ib in range(len(B)):
                        lag = A[ia][0] - B[ib][0]
                        if not (0.5 <= abs(lag) <= 4.0):   # 复调里后继声部常隔 2-4 拍进入
                            continue
                        k = 0
                        while (ia + k < len(A) and ib + k < len(B)
                               and A[ia + k][1] == B[ib + k][1]
                               and abs(A[ia + k][2] - B[ib + k][2]) <= 0.25
                               and abs((A[ia + k][0] - B[ib + k][0]) - lag) <= 0.5):
                            k += 1
                        best = max(best, k)
    return best >= min_run, best


def phrase_spans(slices: list[dict], phrases: list[dict]) -> list[tuple[int, int]]:
    """乐句 → 切片区间 [(start_slice, end_slice)]（含端点）。"""
    ends = [p['start_slice'] for p in phrases]
    out, s = [], 0
    for e in ends:
        out.append((s, e))
        s = e + 1
    return out


def texture_features(slices, events, sp: tuple[int, int]) -> dict:
    a, b = sp
    seg = slices[a:b + 1]
    if not seg:
        return {}
    att_sets = {v: set() for v in VOICES}
    for s in seg:
        for v in VOICES:
            if s['attack'][v]:
                att_sets[v].add(s['off'])
    n_att = [sum(1 for v in VOICES if s['attack'][v]) for s in seg]
    coincidence = sum(1 for k in n_att if k >= 3) / len(seg)
    jac = []
    for i, va in enumerate(VOICES):
        for vb in VOICES[i + 1:]:
            A, B = att_sets[va], att_sets[vb]
            jac.append(len(A & B) / max(len(A | B), 1))
    independence = 1 - (statistics.mean(jac) if jac else 0)
    # 持续音(pedal point): **最低声部**长音保持, 期间上方声部 ≥3 次起音
    t0, t1 = seg[0]['off'], seg[-1]['off'] + seg[-1]['dur']
    pedal_dur = 0.0
    bass_events = events[VOICES[-1]]
    for (st, en, p) in bass_events:
        hold = min(en, t1) - max(st, t0)
        if hold < 2.0:
            continue
        others = sum(1 for v in VOICES[:-1]
                     for (s2, e2, _) in events[v] if max(st, t0) - 1e-6 <= s2 < min(en, t1) - 1e-6)
        if others >= 3:
            pedal_dur += hold
    uni = sum(1 for s in seg if s.get('uniform_pc')) / len(seg)
    return {
        'coincidence': round(coincidence, 3),
        'independence': round(independence, 3),
        'uniform_ratio': round(uni, 3),
        'pedal_ratio': round(pedal_dur / max(t1 - t0, 1e-6), 3),
        'mean_voices': round(statistics.mean([s['n_voices'] for s in seg]), 2),
        'n_slices': len(seg),
    }


def label_texture(feat: dict, imit: bool) -> str:
    """特征 → 织体类型（规则，可复现；阈值经人工抽查 + Ollama 审计校准）。

    优先级: pedal > unison_octave > imitative > chordal > polyphonic > mixed。

    pedal 必须同时满足"低音长音主导"**且**"上方声部呈柱式(同起音率高)"：
    只按住"低音长音 + 上方声部在动"会在文艺复兴复调上大面积误判
    （Palestrina 里低音长音是常态，实测 68% 被误标成 pedal）。
    """
    if not feat:
        return 'unknown'
    if feat['pedal_ratio'] >= 0.45 and feat['coincidence'] >= 0.50:
        return 'pedal'
    if feat['uniform_ratio'] >= 0.40:
        return 'unison_octave'
    if imit:
        return 'imitative'
    if feat['coincidence'] >= 0.55:
        return 'chordal'
    if feat['independence'] >= 0.55:
        return 'polyphonic'
    return 'mixed'


def measure_grid(ch_t, music_end: float | None = None) -> list[tuple[int, float, float]]:
    """真实小节网格 [(index, start, end)]。

    为什么不用"每 4 拍一小节"：众赞歌里有 3/4 拍的作品，而且 music21 的众赞歌
    编码常在末尾带**空白小节**（乐谱小节数 > 实际音乐长度）。用乐谱自己的
    Measure 对象取边界，并按音乐结束位置剔除尾部空小节。
    """
    ms = None
    for p in ch_t.parts:
        found = list(p.getElementsByClass(stream.Measure))
        if found:
            ms = found
            break
    if not ms:
        return []
    grid = []
    for i, m in enumerate(ms):
        st = float(m.getOffsetInHierarchy(ch_t))
        en = st + float(m.barDuration.quarterLength)
        grid.append((i, st, en))
    if music_end is not None:
        grid = [g for g in grid if g[1] < music_end - 1e-6] or grid[:1]
    return grid


def bar_figures(slices, events, grid: list[tuple[int, float, float]]) -> list[dict]:
    """逐小节判定伴奏音型（在最低发声音部上判定）。

    优先级: block（柱式，≥3 声部同起音主导）> alberti > broken > arpeggio
    > sustained > moving。音型判定要求音高落在该小节的和弦音上（用切片级和声），
    避免把级进的旋律音误判成琶音；block/alberti/broken/arpeggio 是为后续
    钢琴语料（莫扎特/贝多芬）预留的，众赞歌里以 block/sustained/moving 为主。
    """
    out = []
    for bar, t0, t1 in grid:
        seg = [s for s in slices if t0 - 1e-6 <= s['off'] < t1 - 1e-6]
        if not seg:
            # 整小节无起音: 若有音在延续 (长和弦/长音跨小节) → sustained, 否则真空白
            sounding = [s for s in slices
                        if s['off'] < t1 - 1e-6 and s['off'] + s['dur'] > t0 + 1e-6
                        and s['n_voices'] > 0]
            out.append({'bar': bar, 'figure': 'sustained' if sounding else 'none',
                        'n_notes': 0})
            continue
        n_att = [sum(1 for v in VOICES if s['attack'][v]) for s in seg]
        block_ratio = sum(1 for k in n_att if k >= 3) / len(seg)
        low = []
        for s in seg:
            ps = [(m, v) for v, m in s['midi'].items() if m is not None]
            if ps:
                m, v = min(ps)
                low.append((s['off'], m, s['dur'], s['attack'][v]))
        attacks = [(o, p, d) for (o, p, d, a) in low if a]
        chord_pcs = set()
        for s in seg:
            if s.get('root') is not None:
                for iv in CHORD_INTERVALS.get(s.get('type') or 'M', [0, 4, 7]):
                    chord_pcs.add((s['root'] + iv) % 12)

        def on_chord(pcs):
            return bool(chord_pcs) and all(p in chord_pcs for p in pcs)

        fig = None
        if block_ratio >= 0.6:
            fig = 'block'
        elif len(attacks) >= 4:
            pcs = [p % 12 for _, p, _ in attacks]
            if on_chord(pcs):
                if (len(pcs) >= 6 and len(set(pcs[:4])) == 3
                        and all(pcs[i] == pcs[i % 4] for i in range(4, len(pcs)))):
                    fig = 'alberti'
                elif len({pcs[i] for i in range(0, len(pcs), 2)}) == 1 and len(set(pcs)) >= 3:
                    fig = 'broken'
                else:
                    ivs = [attacks[i + 1][1] - attacks[i][1] for i in range(len(attacks) - 1)]
                    run = best_run = 1
                    run_ivs = []
                    for i in range(1, len(ivs)):
                        if ivs[i] != 0 and (ivs[i] > 0) == (ivs[i - 1] > 0):
                            run += 1
                            if run == 2:
                                run_ivs = [abs(ivs[i - 1])]
                            run_ivs.append(abs(ivs[i]))
                            if run > best_run:
                                best_run, best_ivs = run, list(run_ivs)
                        else:
                            run = 1
                            run_ivs = []
                    span = max(p for _, p, _ in attacks) - min(p for _, p, _ in attacks)
                    # 琶音: 同向跑动 ≥3 步、音程都是"跳进"(≥2 半音, 不是级进旋律)、
                    # 且跨度 ≥7 半音(五度) —— 众赞歌低音很少有真正的琶音, 该规则主要服务钢琴语料
                    if best_run >= 3 and span >= 7 and all(v >= 2 for v in best_ivs):
                        fig = 'arpeggio'
        if fig is None:
            if not attacks:
                fig = 'sustained' if low else 'none'
            elif len(attacks) == 1:
                fig = 'sustained' if attacks[0][2] >= (t1 - t0) * 0.75 else 'moving'
            else:
                fig = 'moving'
        out.append({'bar': bar, 'figure': fig, 'n_notes': len(attacks)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def build_piece_v2(idx: int, ch) -> dict | None:
    base = build_piece(idx, ch)                 # v1 口径: 调性/和声/乐句/终止式
    if base is None:
        return None
    voices = extract_voices(ch)
    if voices is None:
        return None
    # 与 v1 相同的移调 (大调→C, 小调→c)
    k = ch.analyze('key')
    shift = (0 - k.tonic.pitchClass) % 12
    ch_t = ch.transpose(shift)
    voices_t = extract_voices(ch_t)
    if voices_t is None:
        return None
    events = {v: part_events(voices_t[v]) for v in VOICES}
    if any(len(events[v]) == 0 for v in VOICES):
        return None
    slices = build_slices(events)

    # 切片级和声 (与 v1 同口径: chordify + 罗马数字)
    k2 = ch_t.analyze('key')
    cf = ch_t.chordify()
    harm = []
    for c in cf.flat.notes:
        if not isinstance(c, m21chord.Chord):
            continue
        try:
            rn = roman.romanNumeralFromChord(c, k2)
            deg = rn.scaleDegree
            if deg is None or deg < 1 or deg > 7:
                continue
            ct = rn_to_type(rn)
            if ct is None:
                continue
            harm.append((float(c.getOffsetInHierarchy(cf)), DEGREE_FUNC[deg], ct,
                         rn.root().pitchClass, deg))
        except Exception:
            continue
    harm.sort(key=lambda x: x[0])
    annotate_slices(slices, harm)

    # 乐句: v1 的音符索引 → 切片索引（按 onset 对齐）
    off2slice = {round(s['off'], 3): i for i, s in enumerate(slices)}
    phrases = []
    for ph in base['phrases']:
        off = round(float(base['notes'][ph['note']]['offset']), 3)
        j = off2slice.get(off)
        if j is None:                            # 容差 1e-2 再找一次
            cand = [i for i, s in enumerate(slices) if abs(s['off'] - off) < 1e-2]
            j = cand[0] if cand else None
        if j is None:
            continue
        # 注意: base['phrases'][i]['note'] 是乐句**末音**的音符序号, 这里先记下
        # 该末音所在切片 (`end_slice`), 真正的 start_slice 由 phrase_spans 推出
        phrases.append({'end_slice': j, 'cadence': ph['cadence'],
                        'soprano_note_index': ph['note']})
    if len(phrases) < 2:
        return None

    spans = []
    s = 0
    for p in phrases:
        spans.append((s, p['end_slice']))       # (起始切片, 末切片)
        s = p['end_slice'] + 1

    feat, tex = {}, {}
    for pi, (a, b) in enumerate(spans):
        t0 = slices[a]['off']
        t1 = slices[b]['off'] + slices[b]['dur']
        f = texture_features(slices, events, (a, b))
        if not f:                                # 空区间: 全零特征 + unknown
            f = {'coincidence': 0.0, 'independence': 0.0, 'uniform_ratio': 0.0,
                 'pedal_ratio': 0.0, 'mean_voices': 0.0, 'n_slices': 0}
            imit, run = False, 0
        else:
            imit, run = imitation_score(events, [(t0, t1)])
        f['imitation_run'] = run
        feat[pi] = f
        tex[pi] = label_texture(f, imit)
    music_end = slices[-1]['off'] + slices[-1]['dur']
    grid = measure_grid(ch_t, music_end)
    if not grid:                              # 兜底: 按 4 拍切
        grid = [(i, i * 4.0, (i + 1) * 4.0) for i in range(base['n_measures'])]
    ts = None
    for t in ch_t.parts[0].flat.getElementsByClass(m21meter.TimeSignature):
        ts = t.ratioString          # '3/4' 而不是 str(t) 的 '<music21... 3/4>'
        break
    return {
        'id': base['id'], 'title': base['title'], 'key_original': base['key_original'],
        'style': 'chorale', 'time_signature': ts,
        'n_measures': len(grid), 'voice_order': VOICES,
        'slices': slices,
        'phrases': [{'start_slice': a, 'end_slice': b,
                     'cadence': phrases[i]['cadence'],
                     'soprano_note_index': phrases[i]['soprano_note_index'],
                     'texture': tex[i], 'features': feat[i]}
                    for i, (a, b) in enumerate(spans)],
        'bars': bar_figures(slices, events, grid),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', type=str, default='data/processed/chorales_satb_v2.json')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    pieces, failed = [], 0
    for i, ch in enumerate(corpus.chorales.Iterator()):
        if args.limit and i >= args.limit:
            break
        try:
            p = build_piece_v2(i, ch)
        except Exception as e:
            if args.debug:
                import traceback
                traceback.print_exc()
            p = None
        if p:
            pieces.append(p)
        else:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f'  处理 {i+1}, 成功 {len(pieces)}, 失败 {failed}', flush=True)

    out = ROOT / args.out
    json.dump(pieces, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'完成: {len(pieces)} 首 → {out} (失败 {failed})')

    # ── 统计 ──
    n_slices = [len(p['slices']) for p in pieces]
    tex = Counter(ph['texture'] for p in pieces for ph in p['phrases'])
    fig = Counter(b['figure'] for p in pieces for b in p['bars'])
    cad = Counter(ph['cadence'] for p in pieces for ph in p['phrases'])
    coinc = [ph['features'].get('coincidence', 0) for p in pieces for ph in p['phrases']]
    nv = [s['n_voices'] for p in pieces for s in p['slices']]
    stats = {
        'n_pieces': len(pieces), 'failed': failed,
        'slices_total': sum(n_slices),
        'slices_per_piece_mean': round(statistics.mean(n_slices), 1) if n_slices else 0,
        'texture_dist': dict(tex), 'figure_dist': dict(fig), 'cadence_dist': dict(cad),
        'coincidence_mean': round(statistics.mean(coinc), 3) if coinc else 0,
        'n_voices_mean': round(statistics.mean(nv), 2) if nv else 0,
    }
    json.dump(stats, open(ROOT / 'data/processed/chorales_satb_v2_stats.json', 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n=== 统计 ===')
    for k, v in stats.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
