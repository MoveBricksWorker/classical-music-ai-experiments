"""
Palestrina 复调语料抽取（4 声部，与 SATB 众赞歌**同 schema**）—— 对位预训练数据。

为什么需要
------------------------------------------------
众赞歌四声部也只有 ~74k 音符，且以柱式和弦为主 —— 模型学不到"声部独立进行"。
music21 自带 Palestrina 全集的 **1318 个 .krn 文件（约 809k 音符，平均 5 声部）**，
是免费、离线、无版权风险（19 世纪全集版）的文艺复兴复调语料，正好补这一课。

设计
------------------------------------------------
- 取四个主声部 (Cantus/Altus/Tenor/Bassus)，丢弃 Quintus；
  保持与 SATB 众赞歌**完全相同的字段结构**（slices/phrases/bars/texture/features），
  这样同一套模型可以跨语料预训练 + 微调，只用一个 style 标记区分；
- 与众赞歌的差异（字段层面）：无 func/type/root（文艺复兴调式音乐，罗马数字分析
  无意义）；新增 `style='palestrina'` 与 `voice_names`（原始声部名）；
- 音高统一移调，使**终止和声的最低音 = C**（与众赞歌"全部移调到 C"同思路）；
- 无终止式可标（不使用功能和声概念）：`phrases` 由**全声部休止**切出的段落
  充当（`cadence=None`），这是弥撒/经文歌的乐段结构，不是终止式。

用法
------------------------------------------------
    python build_palestrina_dataset.py [--limit N] [--min-slices 8] [--out PATH]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import statistics
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import music21
from music21 import corpus

from build_chorale_satb_dataset import (VOICES, annotate_slices, bar_figures,
                                        build_slices, extract_voices, imitation_score,
                                        label_texture, measure_grid, part_events,
                                        texture_features)

PAL_DIR = Path(music21.__file__).parent / 'corpus' / 'palestrina'


def list_files(limit: int = 0) -> list[str]:
    files = sorted(f for f in glob.glob(str(PAL_DIR / '**' / '*'), recursive=True)
                   if os.path.isfile(f) and f.endswith('.krn'))
    return files[:limit] if limit else files


def final_root_pc(slices: list[dict]) -> int | None:
    """终止和声的最低音 → 音级（移调基准）。"""
    for s in reversed(slices):
        ps = [m for m in s['midi'].values() if m is not None]
        if ps:
            return min(ps) % 12
    return None


def section_spans(slices: list[dict], min_slices: int = 8,
                  fallback_win: int = 40) -> list[tuple[int, int]]:
    """乐段划分：先用"全声部同时休止"（弥撒/经文歌的段落结构）；
    若几乎没有同时休止（复调各声部轮流呼吸），退化为固定长度窗口。"""
    spans, start = [], 0
    for i, s in enumerate(slices):
        if s['n_voices'] == 0:
            if i - start >= min_slices:
                spans.append((start, i - 1))
            start = i + 1
    if len(slices) - start >= min_slices:
        spans.append((start, len(slices) - 1))
    if len(spans) <= 1 and len(slices) > fallback_win:
        # 几乎没有同时休止 → 退化为固定窗口, 使织体标签是**局部段落**级
        spans = [(i, min(i + fallback_win - 1, len(slices) - 1))
                 for i in range(0, len(slices), fallback_win)]
        spans = [(a, b) for a, b in spans if b - a + 1 >= min_slices]
    return spans or [(0, len(slices) - 1)]


def build_piece(idx: int, path: str, min_slices: int) -> dict | None:
    sc = corpus.parse(path)
    voices = extract_voices(sc)
    if voices is None:
        return None
    # 终止和声最低音 → C
    events0 = {v: part_events(voices[v]) for v in VOICES}
    if any(not events0[v] for v in VOICES):
        return None
    slices0 = build_slices(events0)
    annotate_slices(slices0, [])          # 补 n_voices/uniform_pc 等切片字段
    root = final_root_pc(slices0)
    if root is None:
        return None
    if root != 0:
        sc = sc.transpose((0 - root) % 12)
        voices = extract_voices(sc)
        if voices is None:
            return None
    events = {v: part_events(voices[v]) for v in VOICES}
    if any(len(events[v]) < 8 for v in VOICES):
        return None
    slices = build_slices(events)
    annotate_slices(slices, [])            # 调式音乐无功能和声标注
    spans = section_spans(slices, min_slices)

    phrases = []
    for pi, (a, b) in enumerate(spans):
        f = texture_features(slices, events, (a, b))
        if not f or f['n_slices'] == 0:
            continue
        t0, t1 = slices[a]['off'], slices[b]['off'] + slices[b]['dur']
        imit, run = imitation_score(events, [(t0, t1)])
        f['imitation_run'] = run
        phrases.append({'start_slice': a, 'end_slice': b, 'cadence': None,
                        'texture': label_texture(f, imit), 'features': f})
    if not phrases:
        return None
    music_end = slices[-1]['off'] + slices[-1]['dur']
    grid = measure_grid(sc, music_end)
    if not grid:
        grid = [(i, i * 4.0, (i + 1) * 4.0) for i in range(max(1, int(music_end) // 4))]
    return {
        'id': f'pal_{idx:04d}',
        'title': (sc.metadata.title if sc.metadata and sc.metadata.title
                  else os.path.basename(path)),
        'source': os.path.relpath(path, PAL_DIR),
        'style': 'palestrina',
        'voice_names': {v: (voices[v].partName or voices[v].id or v) for v in VOICES},
        'key_original': None,
        'n_measures': len(grid),
        'voice_order': VOICES,
        'slices': slices,
        'phrases': phrases,
        'bars': bar_figures(slices, events, grid),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-slices', type=int, default=8)
    ap.add_argument('--out', type=str, default='data/processed/palestrina_satb_v1.json')
    ap.add_argument('--gzip', action='store_true', help='输出 .json.gz (体积约 1/5)')
    ap.add_argument('--debug', action='store_true', help='打印失败原因')
    args = ap.parse_args()

    files = list_files(args.limit)
    print(f'Palestrina 文件 {len(files)} 个 (目录 {PAL_DIR})')
    pieces, failed = [], 0
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            p = build_piece(i, f, args.min_slices)
        except Exception as e:
            if args.debug:
                import traceback
                print(f'  [失败] {os.path.basename(f)}: {type(e).__name__}: {e}')
                traceback.print_exc()
            p = None
        if p:
            pieces.append(p)
        else:
            failed += 1
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(files)} | 成功 {len(pieces)} 失败 {failed} | '
                  f'{time.time()-t0:.0f}s', flush=True)

    out = ROOT / args.out
    if args.gzip:
        with gzip.open(str(out) + '.gz', 'wt', encoding='utf-8') as fh:
            json.dump(pieces, fh, ensure_ascii=False)
        size = os.path.getsize(str(out) + '.gz') / 1e6
        print(f'完成: {len(pieces)} 首 → {out}.gz ({size:.1f} MB)')
    else:
        json.dump(pieces, open(out, 'w', encoding='utf-8'), ensure_ascii=False)
        print(f'完成: {len(pieces)} 首 → {out} '
              f'({os.path.getsize(out)/1e6:.1f} MB)')

    n_slices = [len(p['slices']) for p in pieces]
    tex = Counter(ph['texture'] for p in pieces for ph in p['phrases'])
    fig = Counter(b['figure'] for p in pieces for b in p['bars'])
    stats = {
        'n_pieces': len(pieces), 'failed': failed,
        'slices_total': sum(n_slices),
        'notes_est': sum(s['n_voices'] for p in pieces for s in p['slices']),
        'slices_per_piece_mean': round(statistics.mean(n_slices), 1) if n_slices else 0,
        'texture_dist': dict(tex), 'figure_dist': dict(fig),
        'secs': round(time.time() - t0, 1),
    }
    json.dump(stats, open(ROOT / 'data/processed/palestrina_satb_v1_stats.json', 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=2)
    print('=== 统计 ===')
    for k, v in stats.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
