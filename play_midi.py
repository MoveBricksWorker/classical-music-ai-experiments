"""播放 MIDI 文件（用 music21 的实时播放器，走系统软音源）。

    python play_midi.py                       # 播放下面 RECOMMENDED 里列出的成品
    python play_midi.py 某文件.mid [另一个.mid]
    python play_midi.py --seconds 20 某文件.mid   # 只试听 20 秒

实时播放依赖 pygame.mixer 的 MIDI 支持（Windows 上走系统自带的 GS 软音源）。
若不可用，自动退回 `stream.show('midi')`，交给系统默认播放器打开。
"""
import argparse
import sys
import time
import types
from pathlib import Path

# music21 v9 的 StreamPlayer 内部 `import pygame.exceptions` 并要求
# PygameError / get_error()，而 pygame 2.6 已没有该子模块 → 先补一个兼容层，
# 否则它会误报 "StreamPlayer requires pygame. Install first"。
try:
    import pygame.exceptions  # noqa: F401
except ImportError:
    import pygame

    _shim = types.ModuleType('pygame.exceptions')
    _shim.PygameError = pygame.error
    _shim.get_error = getattr(pygame, 'get_error', lambda: '')
    sys.modules['pygame.exceptions'] = _shim
    pygame.exceptions = _shim

from music21 import converter                      # noqa: E402
from music21.midi.realtime import StreamPlayer     # noqa: E402

# 默认播放清单：新网格的端到端成品 + 配和声 A/B（真巴赫 vs 模型）
RECOMMENDED = [
    'data/generated/full_modelgrid/full_1.mid',
    'data/generated/satb_chorale_50_ref.mid',
    'data/generated/satb_chorale_50_gen.mid',
]


def play(path: Path, seconds: float | None = None) -> None:
    score = converter.parse(str(path))
    try:
        player = StreamPlayer(score)
        print(f'▶ {path.name}（播放中…{"%g 秒后停" % seconds if seconds else "Ctrl+C 可中断"}）')
        player.play(playForMilliseconds=(seconds * 1000 if seconds else float('inf')))
    except Exception as exc:                                   # pygame 不可用/无设备
        print(f'  实时播放不可用（{type(exc).__name__}: {exc}）→ 交给系统默认播放器')
        score.show('midi')
        time.sleep(1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mids', nargs='*', help='MIDI 路径；不给则播放默认清单')
    ap.add_argument('--seconds', type=float, default=None, help='每首只听这么多秒')
    args = ap.parse_args()

    root = Path(__file__).parent
    todo = args.mids or [str(root / r) for r in RECOMMENDED]
    for m in todo:
        p = Path(m)
        if not p.exists():
            print(f'找不到 {p}', file=sys.stderr)
            continue
        try:
            play(p, args.seconds)
        except KeyboardInterrupt:
            print('\n已中断')
            return
    print('播放结束')


if __name__ == '__main__':
    main()
