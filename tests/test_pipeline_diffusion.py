"""
管线集成测试 (不依赖 578MB 和弦权重): 用固定和弦进行走通
"扩散旋律生成 → 乐理精修 → 织体 → 三轨 MIDI" 完整链路。

用法: python tests/test_pipeline_diffusion.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import torch

from gen_full import (load_diffusion_model, generate_midi, polish_melody,
                      score_candidate, split_phrases, DIFFUSION_PT)
from constants import CHORD_INTERVALS, F2ID_MELODY, T2ID
from model.architectures import chord_name_to_info


def make_chords(n=12):
    """合成 C 大调功能进行: T - PD - D - T 循环。"""
    pattern = [
        {'func': 'T', 'chord': 'I', 'dur': 3, 'beat': 1, 'inv': 0, 'cad': 0},
        {'func': 'PD', 'chord': 'IV', 'dur': 3, 'beat': 2, 'inv': 0, 'cad': 0},
        {'func': 'D', 'chord': 'V7', 'dur': 3, 'beat': 3, 'inv': 0, 'cad': 0},
        {'func': 'T', 'chord': 'I', 'dur': 7, 'beat': 1, 'inv': 0, 'cad': 0},
    ]
    return [dict(c) for _ in range(n // 4) for c in pattern]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device} | 扩散权重: {DIFFUSION_PT}')

    model = load_diffusion_model(device)
    chords = make_chords(48)
    n_chords = len(chords)
    total_notes = n_chords * 4

    # 条件: 每个和弦 4 音
    cf_t, ct_t, cr_t = [], [], []
    for c in chords:
        r, ct = chord_name_to_info(c['chord'])
        fid = min(F2ID_MELODY.get(c['func'], 4), len(F2ID_MELODY) - 1)
        tid = min(T2ID.get(ct, 0), len(T2ID) - 1)
        cf_t += [fid] * 4
        ct_t += [tid] * 4
        cr_t += [r] * 4
    cf = torch.tensor([cf_t], device=device)
    ct = torch.tensor([ct_t], device=device)
    cr = torch.tensor([cr_t], device=device)

    chord_tones_all = []
    for c in chords:
        r, ct_ = chord_name_to_info(c['chord'])
        chord_tones_all.append([(r + iv) % 12 for iv in CHORD_INTERVALS.get(ct_, [0, 4, 7])])

    def melody_scorer(pc, cts, prev_pc, step):
        s = 0.0
        if pc in cts:
            s += 0.3
        dist = min(abs(pc - prev_pc), 12 - abs(pc - prev_pc))
        if 1 <= dist <= 2:
            s += 0.4
        elif dist >= 6:
            s -= 0.3
        if pc == prev_pc:
            s -= 0.3
        return s

    # 3 候选 + 择优 (与 gen_full 同口径)
    best = None
    best_score = -999
    for cand in range(3):
        gp, grh = model.generate(cf[:, ::4], ct[:, ::4], cr[:, ::4],
                                 steps=16, temp=0.85, remask_steps=6,
                                 remask_ratio=0.3, scorer=melody_scorer,
                                 scorer_steps=4, seed=cand * 1000 + 7)
        pcs = [int(p) for p in gp[0].tolist()]
        rhs = [int(r) for r in grh[0].tolist()]
        s = score_candidate(pcs, chord_tones_all)
        if s > best_score:
            best_score, best = s, (pcs, rhs)
    print(f'最优候选得分: {best_score:.2f}')

    m_pcs, m_rhythms = best
    midi_pitches, m_rhythms = polish_melody(m_pcs, m_rhythms, chords, chord_tones_all)

    out = ROOT / 'data/generated/pipeline_integration_test.mid'
    generate_midi(chords, midi_pitches, m_rhythms, bpm=92, output_path=str(out))

    # 验证 MIDI 可解析且三轨齐全
    import mido
    mid = mido.MidiFile(str(out))
    assert len(mid.tracks) == 4, f'应为 4 轨 (tempo + 3), 实际 {len(mid.tracks)}'
    n_notes = sum(1 for t in mid.tracks[1:] for m in t if m.type == 'note_on' and m.velocity > 0)
    print(f'MIDI 验证通过: 4 轨, {n_notes} 个音符起音 → {out}')
    print(f'乐句数: {len(split_phrases(chords))}')


if __name__ == '__main__':
    main()
