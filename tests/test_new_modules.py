"""新模块冒烟测试: 乐理指标 / 防模仿指标 / 扩散模型核心行为。

用法: python tests/test_new_modules.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from metrics.theory_metrics import (tokens_to_notes, evaluate_tokens,
                                    melodic_smoothness, rhythm_matching, key_confidence)
from metrics.anti_imitation import (trigram_jaccard, melody_features,
                                    mahalanobis, simaa)
from model.melody_diffusion import MelodyDiffusion, PITCH_MASK, RHYTHM_MASK


def test_smoothness():
    # 纯级进旋律: step_ratio 应为 1.0
    sm = melodic_smoothness(tokens_to_notes([0, 1, 2, 3, 4, 3, 2, 1], [4] * 8))
    assert abs(sm['step_ratio'] - 1.0) < 1e-6
    # 大三全音跳 (pc 距离 6): leap_ratio 应为 1.0
    sm2 = melodic_smoothness(tokens_to_notes([0, 6, 0, 6, 0, 6, 0, 6], [4] * 8))
    assert abs(sm2['leap_ratio'] - 1.0) < 1e-6
    print('smoothness OK')


def test_rhythm():
    # 全四分音符: 强拍起音率 50%, 强拍空拍率 ~0
    rm = rhythm_matching(tokens_to_notes([0] * 32, [4] * 32))
    assert abs(rm['strong_onset_ratio'] - 0.5) < 1e-6
    assert rm['strong_beat_gap_ratio'] < 0.1
    print('rhythm OK')


def test_key():
    # C 大调音阶 (C E G 为主): 应判定 C major 且置信度 > 0.7
    notes = tokens_to_notes([0, 4, 7, 0, 7, 4, 2, 5, 9, 7, 4, 0, 7, 0, 4, 7], [4] * 16)
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += 1
    k = key_confidence(hist)
    assert k['key'] == 'C major', k['key']
    assert k['confidence'] > 0.7, k['confidence']
    print('key OK:', k['key'], round(k['confidence'], 3))


def test_simaa_identity():
    # 与训练集完全相同的生成序列: SIMAA@90 应为 1.0
    train = [0, 2, 4, 5, 7, 9, 11, 9, 7, 5, 4, 2, 0, 2, 4, 5, 7, 9, 11, 9]
    s = simaa(train, [train[:16]])
    assert s['SIMAA@90'] == 1.0 and s['mean_max_sim'] == 1.0
    print('simaa OK')


def test_mahalanobis_self():
    pieces = [([i % 12 for i in range(j, j + 16)], [4] * 16) for j in range(30)]
    rep = mahalanobis(pieces, pieces)
    assert abs(rep['t_stat']) < 1e-6 and abs(rep['p_value'] - 1.0) < 1e-6
    print('mahalanobis OK: train mean %.2f ± %.2f' % (rep['train_mean'], rep['train_std']))


def test_features_17d():
    f = melody_features([0, 2, 4, 5, 7, 9, 11, 9, 7, 5, 4, 2, 0, 2, 4, 5], [4] * 16)
    assert f.shape == (17,)
    assert abs(f[:12].sum() - 1.0) < 1e-6
    print('features OK')


def test_diffusion():
    torch.manual_seed(0)
    m = MelodyDiffusion(d=60, h=4, L=2, max_len=64)
    B, L = 2, 16
    pitch = torch.randint(0, 13, (B, L))
    rhythm = torch.randint(0, 8, (B, L))
    func = torch.randint(0, 7, (B, L))
    typ = torch.randint(0, 9, (B, L))
    root = torch.randint(0, 12, (B, L))

    # 训练损失有限
    loss = m.training_loss(pitch, rhythm, func, typ, root)[0]
    assert np.isfinite(loss.item())

    # 全掩码生成: 无 MASK 残留, 数值合法
    gp, grh = m.generate(func[:1, :4], typ[:1, :4], root[:1, :4],
                         steps=8, temp=1.2, seed=1)
    assert (gp < PITCH_MASK).all() and (grh < RHYTHM_MASK).all()
    assert gp.shape == (1, 16)

    # 局部重采样: 区间外不变
    p = torch.randint(0, 13, (1, 16)); r = torch.randint(0, 8, (1, 16))
    p2, r2 = m.resample(p, r, func[:1], typ[:1], root[:1], 4, 8, steps=4)
    assert (p[:, :4] == p2[:, :4]).all() and (p[:, 8:] == p2[:, 8:]).all()
    print('diffusion OK')


if __name__ == '__main__':
    test_smoothness()
    test_rhythm()
    test_key()
    test_simaa_identity()
    test_mahalanobis_self()
    test_features_17d()
    test_diffusion()
    print('全部测试通过 ✔')
