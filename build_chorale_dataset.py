"""
构建巴赫众赞歌句子级数据集 (步骤 1/2)。

与旧语料的本质区别: 众赞歌是**完整作品** (12-20 小节, 4-6 个乐句),
乐句边界由**终止式** (理论真值) 界定, 而非"换气直觉"。

流程 (每首):
    1. 调性分析 → 移调到 C (保持调式, 大调→C大调, 小调→c小调);
    2. 提取 Soprano 旋律 (我们的旋律数据格式: pc + rhythm bin);
    3. chordify + 罗马数字分析 → 逐音和声 (func/type/root);
    4. 终止式检测 (V(7)→I / V→vi / IV→I / 半终止) + 贪心分段 → 乐句;
    5. 输出 JSON: 音符 + 乐句 + 终止式类型 + 小节数。

用法: python build_chorale_dataset.py [--limit N] [--out PATH]
"""
from __future__ import annotations
import sys, os, json, argparse, warnings
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')

from music21 import corpus, chord as m21chord, roman, key as m21key, note as m21note, stream

ROOT = os.path.dirname(os.path.abspath(__file__))

MELODY_RHYTHM = {0: 0.125, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.5, 6: 2.0, 7: 4.0}

# 功能映射 (教科书): 1/3/6→T, 2/4→PD, 5/7→D
DEGREE_FUNC = {1: 'T', 2: 'PD', 3: 'T', 4: 'PD', 5: 'D', 6: 'T', 7: 'D'}


def dur_to_bin(ql: float) -> int:
    """quarterLength → 我们的 8 类节奏 bin。"""
    bins = sorted(MELODY_RHYTHM.items(), key=lambda kv: kv[1])
    best, bd = 4, 1e9
    for k, v in bins:
        if abs(ql - v) < bd:
            bd, best = abs(ql - v), k
    return best


def rn_to_type(rn) -> str | None:
    """music21 罗马数字 → 我们的和弦类型名。"""
    q = rn.quality
    sev = rn.seventh
    if sev is not None:
        if q == 'major':
            return 'M7' if sev == 'major' else 'dom7'
        if q == 'minor':
            return 'm7'
        if q == 'diminished':
            return 'hdim7' if sev == 'minor' else 'dim7'
    return {'major': 'M', 'minor': 'm', 'diminished': 'dim', 'augmented': 'aug'}.get(q)


def cadence_of(prev_deg, prev_type, cur_deg, cur_type) -> str | None:
    """相邻两个和声事件构成终止式吗 (返回类型)。"""
    # 属功能和弦: 5 或 7(导音), 含七和弦
    is_dom = prev_deg in (5, 7)
    if is_dom and cur_deg == 1:
        return 'authentic'
    if is_dom and cur_deg == 6:
        return 'deceptive'
    if prev_deg == 4 and cur_deg == 1:
        return 'plagal'
    return None


def build_piece(idx: int, ch) -> dict | None:
    try:
        k = ch.analyze('key')
        if k.mode not in ('major', 'minor'):
            return None
        shift = (0 - k.tonic.pitchClass) % 12
        ch_t = ch.transpose(shift)
        k2 = ch_t.analyze('key')
        # 旋律: Soprano
        sop = None
        for p in ch_t.parts:
            if p.partName and 'oprano' in p.partName:
                sop = p
                break
        if sop is None:
            return None
        mel_notes = [n for n in sop.flat.notes if isinstance(n, m21note.Note)]
        if len(mel_notes) < 12:
            return None
        n_measures = len(ch_t.parts[0].getElementsByClass(stream.Measure))

        # 和声: chordify + 罗马数字 (绝对 offset)
        cf = ch_t.chordify()
        harm_events = []          # (offset, func, type, root_pc, degree, quality)
        for c in cf.flat.notes:
            if not isinstance(c, m21chord.Chord):
                continue
            try:
                rn = roman.romanNumeralFromChord(c, k2)
            except Exception:
                continue
            deg = rn.scaleDegree
            if deg is None or deg < 1 or deg > 7:
                continue
            ct = rn_to_type(rn)
            if ct is None:
                continue
            try:
                root_pc = rn.root().pitchClass
            except Exception:
                continue
            off = float(c.getOffsetInHierarchy(cf))
            harm_events.append((off, DEGREE_FUNC[deg], ct, root_pc, deg, rn.quality))
        if len(harm_events) < 8:
            return None
        harm_events.sort(key=lambda x: x[0])

        # 逐音和声 (取 onset 处正在发声的和声)
        def harm_at(off):
            cur = harm_events[0]
            for h in harm_events:
                if h[0] <= off + 1e-6:
                    cur = h
                else:
                    break
            return cur

        notes = []
        melody_onsets = []
        for n in mel_notes:
            off = float(n.getOffsetInHierarchy(ch_t))
            h = harm_at(off)
            notes.append({
                'pc': n.pitch.pitchClass,
                'rhythm': dur_to_bin(float(n.quarterLength)),
                'offset': round(off, 3),
                'ql': float(n.quarterLength),
                'func': h[1], 'type': h[2], 'root': h[3],
            })
            melody_onsets.append(off)

        # ── 终止式检测: 和声事件序列中的 D→T / D→vi / PD→T ──
        cands = []      # (offset, cadence_type)
        for i in range(1, len(harm_events)):
            p, c = harm_events[i - 1], harm_events[i]
            ct = cadence_of(p[4], p[2], c[4], c[2])
            if ct:
                cands.append((c[0], ct))

        # 映射到旋律音符索引: 终止式落点 → 附近**时值最长**的旋律音
        # (古典乐句的真收束音通常最长; 直接取和声解决点会偏 1-2 音)
        def note_index_at(off, snap_window: int = 3):
            idx = 0
            for j, o in enumerate(melody_onsets):
                if o <= off + 1e-6:
                    idx = j
                else:
                    break
            lo = max(0, idx - snap_window)
            hi = min(len(notes) - 1, idx + snap_window)
            best_j, best_key = idx, None
            for j in range(lo, hi + 1):
                key = (notes[j]['ql'], -abs(j - idx))
                if best_key is None or key > best_key:
                    best_key, best_j = key, j
            return best_j

        raw_bounds = []
        for off, ct in cands:
            j = note_index_at(off)
            raw_bounds.append((j, ct, off))

        # 半终止候选: 旋律长音 (>= 半音符) 且该处为属功能, 强拍 (每小节 1/3 拍)
        for j, nt in enumerate(notes):
            if nt['func'] != 'D' or nt['ql'] < 2.0:
                continue
            if round(nt['offset']) % 4 not in (0, 2):
                continue
            if any(b[0] == j for b in raw_bounds):
                continue
            raw_bounds.append((j, 'half', nt['offset']))
        raw_bounds.sort(key=lambda x: (x[0], {'authentic': 0, 'deceptive': 1,
                                              'plagal': 2, 'half': 3}[x[1]]))

        # ── 贪心分段: 最小乐句 4 音 / 12 拍, 最大 ~24 拍 ──
        MIN_NOTES, MIN_QL, MAX_QL = 3, 8.0, 24.0
        bounds = []
        last_idx, last_off = -1, 0.0
        for j, ct, off in raw_bounds:
            if j <= last_idx:
                continue
            ql_span = melody_onsets[j] - last_off
            if ql_span < MIN_QL or (j - last_idx) < MIN_NOTES:
                continue
            bounds.append({'note': j, 'cadence': ct})
            last_idx, last_off = j, melody_onsets[j]
        # 末端: 最后一个音符必为句末 (全终止)
        if bounds and bounds[-1]['note'] != len(notes) - 1:
            span_ql = melody_onsets[-1] - melody_onsets[bounds[-1]['note']] if bounds else 0
            if span_ql < MIN_QL:
                bounds[-1] = {'note': len(notes) - 1, 'cadence': 'authentic'}
            else:
                bounds.append({'note': len(notes) - 1, 'cadence': 'authentic'})
        if not bounds:
            bounds = [{'note': len(notes) - 1, 'cadence': 'authentic'}]

        # 乐句数至少 2 (整曲是一句 → 太长, 按最长 24 拍切)
        if len(bounds) < 2:
            j = len(notes) - 1
            # 从中间找一个长音处切
            mid = len(notes) // 2
            best_j = max(range(MIN_NOTES, len(notes) - 2),
                         key=lambda t: (notes[t]['ql'], -abs(t - mid)))
            bounds = [{'note': best_j, 'cadence': 'half'},
                      {'note': len(notes) - 1, 'cadence': 'authentic'}]

        return {
            'id': f'chorale_{idx}',
            'title': str(ch.metadata.title) if ch.metadata else '',
            'key_original': str(k),
            'n_measures': n_measures,
            'notes': notes,
            'phrases': bounds,
        }
    except Exception as e:
        if os.environ.get('DEBUG_BUILD'):
            import traceback
            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out', type=str, default='data/processed/chorales_sentences_v1.json')
    args = parser.parse_args()

    pieces, failed = [], 0
    for i, ch in enumerate(corpus.chorales.Iterator()):
        if args.limit and i >= args.limit:
            break
        p = build_piece(i, ch)
        if p:
            pieces.append(p)
        else:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f'  处理 {i+1} 首, 成功 {len(pieces)}, 失败 {failed}')

    out = os.path.join(ROOT, args.out)
    json.dump(pieces, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'完成: {len(pieces)} 首 → {out} (失败 {failed})')

    # 统计
    n_notes = [len(p['notes']) for p in pieces]
    import statistics
    print(f'音符/首: 均值 {statistics.mean(n_notes):.1f} 中位数 {statistics.median(n_notes)} 范围 [{min(n_notes)},{max(n_notes)}]')
    plens, cads, nph = [], Counter(), []
    for p in pieces:
        nph.append(len(p['phrases']))
        prev = -1
        for b in p['phrases']:
            plens.append(b['note'] - prev)
            cads[b['cadence']] += 1
            prev = b['note']
    print(f'乐句数/首: 均值 {statistics.mean(nph):.1f} | 乐句长度(音): 均值 {statistics.mean(plens):.1f} 中位数 {statistics.median(plens)}')
    print(f'终止式类型分布: {dict(cads)}')


if __name__ == '__main__':
    main()
