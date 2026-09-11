"""
DeepBach 式局部重采样演示 (Briot & Pachet #18)。

取语料中一段真实旋律, 把指定小节重新打掩码, 在"上下文 = 其余真值"
条件下用扩散模型重生成 → 输出 原曲 / 重生成 两段可听 MIDI。

用法:
    python demo_resample.py --start 4 --end 12 --steps 8 --out data/generated/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

import argparse, json, random

import torch

from constants import F2ID_MELODY, T2ID
from model.melody_diffusion import MelodyDiffusion
from export_samples import sample_to_midi


def load_window():
    d = json.load(open(ROOT / 'data/processed/melody_llm_full_v2.json', encoding='utf-8'))
    random.seed(7)
    start = random.randrange(0, len(d) - 16)
    chunk = d[start:start + 16]
    return {
        'func': [F2ID_MELODY.get(c.get('func', 'Other'), 4) for c in chunk],
        'type': [T2ID.get(c.get('type', 'M'), 0) for c in chunk],
        'root': [min(11, c['root']) for c in chunk],
        'pc': [min(12, c['pc']) for c in chunk],
        'rhythm': [min(7, c['rhythm']) for c in chunk],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=4, help='重生成区间起点 (音位)')
    parser.add_argument('--end', type=int, default=12, help='重生成区间终点 (音位, 不含)')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--pt', type=str, default='melody_diffusion_v1.pt')
    parser.add_argument('--out', type=str, default='data/generated')
    parser.add_argument('--n-variants', type=int, default=3)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MelodyDiffusion(d=252, h=6, L=8, max_len=256,
                            num_funcs=len(F2ID_MELODY), num_types=len(T2ID)).to(device)
    model.load_state_dict(torch.load(ROOT / args.pt, map_location=device, weights_only=True))
    model.eval()

    s = load_window()
    pitch = torch.tensor([s['pc']], device=device)
    rhythm = torch.tensor([s['rhythm']], device=device)
    func = torch.tensor([s['func']], device=device)
    typ = torch.tensor([s['type']], device=device)
    root = torch.tensor([s['root']], device=device)

    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)

    # 原曲
    sample_to_midi({'pitch': s['pc'], 'rhythm': s['rhythm'],
                    'harmony': [{'root': s['root'][j * 4], 'type_id': s['type'][j * 4]}
                                for j in range(4)]},
                   out_dir / 'resample_original.mid')

    print(f'原曲: {s["pc"]}')
    print(f'重生成区间: [{args.start}, {args.end})  ({args.end - args.start} 音位)')
    for v in range(args.n_variants):
        p2, r2 = model.resample(pitch, rhythm, func, typ, root,
                                args.start, args.end, steps=args.steps, seed=v + 1)
        p2, r2 = p2[0].tolist(), r2[0].tolist()
        outside_ok = (p2[:args.start] == s['pc'][:args.start] and
                      p2[args.end:] == s['pc'][args.end:])
        print(f'变体{v + 1}: {p2}  (区间外不变: {outside_ok})')
        sample_to_midi({'pitch': p2, 'rhythm': r2,
                        'harmony': [{'root': s['root'][j * 4], 'type_id': s['type'][j * 4]}
                                    for j in range(4)]},
                       out_dir / f'resample_variant_{v + 1}.mid')

    print(f'已输出 → {out_dir}/resample_*.mid (原曲 + {args.n_variants} 个变体)')


if __name__ == '__main__':
    main()
