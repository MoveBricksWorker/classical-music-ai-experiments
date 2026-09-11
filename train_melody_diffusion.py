"""
MelodyDiffusion 训练脚本 —— 非自回归离散扩散旋律模型 (GETMusic 式 D3PM)。

用法:
    python train_melody_diffusion.py --epochs 100 --batch 128 --d 252 --layers 8 --steps 16

数据: data/processed/melody_llm_full_v2.json (13,592 音, 窗口 SEQ=16 STRIDE=4)
规模: d=252/8层 ≈ 6.5M 参数, 适配 8GB 消费级 GPU (4060 Laptop)。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

import argparse, json, math, time, random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim

from constants import F2ID_MELODY, T2ID, MELODY_RHYTHM
from model.melody_diffusion import MelodyDiffusion, PITCH_MASK, RHYTHM_MASK, REST


# ═══════════════════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════════════════

SEQ, STRIDE = 16, 4


def compute_beat_positions(notes, beats_per_measure=4):
    """按乐曲 (prev_pc=null 为界) 从第 0 拍累计时值, 推算每音真实拍位 mod 4。"""
    beats = []
    t = 0.0
    for n in notes:
        if n.get('prev_pc') is None:
            t = 0.0
        beats.append(int(t) % beats_per_measure)
        t += MELODY_RHYTHM.get(min(7, n.get('rhythm', 4)), 1.0)
    return beats


def compute_struct_labels(notes, num_pos_bins: int = 8, num_toend_bins: int = 4):
    """从乐曲边界自动推导结构标签 (无需 LLM 重新标注):
        pos_bin   : 全曲相对位置 (0=起始, num_pos_bins-1=结尾)
        toend_bin : 距结尾剩余比例 → 0=结尾处 / 1=收束 / 2=中段 / 3=远离结尾
    """
    pieces, cur = [], []
    for n in notes:
        if n.get('prev_pc') is None and cur:
            pieces.append(cur); cur = []
        cur.append(n)
    if cur:
        pieces.append(cur)
    labels = []
    for p in pieces:
        L = max(len(p), 1)
        for i in range(len(p)):
            pos_bin = min(num_pos_bins - 1, int(i / L * num_pos_bins))
            rem = 1.0 - (i + 1) / L
            if rem < 0.06:
                tb = 0
            elif rem < 0.15:
                tb = 1
            elif rem < 0.40:
                tb = 2
            else:
                tb = 3
            labels.append((pos_bin, min(tb, num_toend_bins - 1)))
    return labels


def load_phrase_labels(phrases_path: str, n_notes: int,
                       num_bins: int = 6) -> list[int]:
    """从乐句标注文件生成 每音 → 距句末音数 bin (0=句末换气点, 封顶 num_bins-1)。

    bin 值 = min(到本句最后一个音的步数, num_bins-1)。
    """
    d = json.load(open(phrases_path, encoding='utf-8'))
    labels = [0] * n_notes
    for p in d['pieces']:
        start, L, breaths = p['start'], p['len'], p['breaths']
        prev = -1
        for b in breaths:
            for i in range(prev + 1, b + 1):
                labels[start + i] = min(b - i, num_bins - 1)
            prev = b
    return labels


def load_melody_windows(json_path: str, with_beat: bool = False,
                        block_conds: bool = False, with_struct: bool = False,
                        with_phrase: bool = False, phrases_path: str = ''):
    """旋律语料 → 16 音窗口 (与 train_big.py load_melody_data 同口径)。

    block_conds=True 时把每 4 音的和声条件改为块内多数投票后重复 4 次
    —— 与推理管线"每和弦 4 个旋律音共用同一条件"的输入模式一致。
    (语料 LLM 标注的和声几乎逐音变化: 连续相同段长度=1 占 98%,
    长度=4 仅 1 例; 训练/推理条件不一致是管线输出振荡/嗡鸣的根因之一,
    原版 MelodyGPT 同样存在该问题。)
    """
    with open(json_path, encoding='utf-8') as f:
        notes = json.load(f)
    all_beats = compute_beat_positions(notes) if with_beat else None
    all_struct = compute_struct_labels(notes) if with_struct else None
    all_phrase = (load_phrase_labels(phrases_path, len(notes))
                  if with_phrase and phrases_path else None)
    samples = []
    for i in range(0, len(notes) - SEQ, STRIDE):
        chunk = notes[i:i + SEQ]
        if len(chunk) < SEQ:
            continue
        funcs = [F2ID_MELODY.get(n.get('func', 'Other'), 4) for n in chunk]
        types = [T2ID.get(n.get('type', 'M'), 0) for n in chunk]
        roots = [min(11, n['root']) for n in chunk]
        if block_conds:
            bf, bt, br = [], [], []
            for j in range(0, SEQ, 4):
                bf += [Counter(funcs[j:j + 4]).most_common(1)[0][0]] * 4
                bt += [Counter(types[j:j + 4]).most_common(1)[0][0]] * 4
                br += [Counter(roots[j:j + 4]).most_common(1)[0][0]] * 4
            funcs, types, roots = bf, bt, br
        s = {
            'func': funcs, 'type': types, 'root': roots,
            'pc': [min(REST, n['pc']) for n in chunk],
            'rhythm': [min(7, n['rhythm']) for n in chunk],
        }
        if with_beat:
            s['beat'] = all_beats[i:i + SEQ]
        if with_struct:
            st = all_struct[i:i + SEQ]
            s['pos_bin'] = [x[0] for x in st]
            s['toend_bin'] = [x[1] for x in st]
        if with_phrase:
            s['phrase_bin'] = all_phrase[i:i + SEQ]
        samples.append(s)
    random.shuffle(samples)
    return samples


class MelodyDiffDataset(Dataset):
    def __init__(self, samples):
        self.s = samples

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        out = [torch.tensor(s['func']), torch.tensor(s['type']),
               torch.tensor(s['root']), torch.tensor(s['pc']),
               torch.tensor(s['rhythm'])]
        if 'beat' in s:
            out.append(torch.tensor(s['beat']))
        if 'pos_bin' in s:
            out.append(torch.tensor(s['pos_bin']))
            out.append(torch.tensor(s['toend_bin']))
        if 'phrase_bin' in s:
            out.append(torch.tensor(s['phrase_bin']))
        return tuple(out)


# ═══════════════════════════════════════════════════════════════
# 评估: 掩码恢复准确率 + 全掩码迭代生成准确率 (GETMusic CA 口径)
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_recovery(model, loader, device, with_beat, with_struct=False, n_batches=10):
    """同分布随机掩码下的恢复准确率 (与训练损失同分布)。"""
    tot_p = tot_r = cor_p = cor_r = 0
    for bi, b in enumerate(loader):
        if bi >= n_batches:
            break
        fn, tp, rt, pc, rh = b[:5]
        idx = 5
        beat = None; pos_bin = None; toend_bin = None; phrase_bin = None
        if with_beat and len(b) > idx:
            beat = b[idx].to(device); idx += 1
        if len(b) > idx + 1:
            pos_bin = b[idx].to(device); toend_bin = b[idx + 1].to(device); idx += 2
        if len(b) > idx:
            phrase_bin = b[idx].to(device)
        fn, tp, rt, pc, rh = [x.to(device) for x in (fn, tp, rt, pc, rh)]
        B, L = pc.shape
        t = torch.empty(B, device=device).uniform_(0.05, 0.95)
        mask = torch.rand(B, L, device=device) < t.unsqueeze(1)
        p_in = pc.clone(); r_in = rh.clone()
        p_in[mask] = PITCH_MASK; r_in[mask] = RHYTHM_MASK
        flag = (~mask).long() * 2 + (~mask).long()
        pl, rl = model(p_in, r_in, fn, tp, rt, flag, beat, pos_bin, toend_bin, phrase_bin)
        pp = pl.argmax(-1); rp = rl.argmax(-1)
        cor_p += (pp[mask] == pc[mask]).sum().item()
        cor_r += (rp[mask] == rh[mask]).sum().item()
        tot_p += mask.sum().item(); tot_r += mask.sum().item()
    return cor_p / max(tot_p, 1), cor_r / max(tot_r, 1)


@torch.no_grad()
def eval_generation(model, samples, device, n=64, steps=16, temp=1.0, with_beat=False):
    """全 [MASK] 出发迭代生成, 与真值对比 (条件=真实和声)。"""
    idxs = random.sample(range(len(samples)), min(n, len(samples)))
    cor_p = cor_r = tot = 0
    t0 = time.time()
    for i in idxs:
        s = samples[i]
        fn = torch.tensor([s['func']], device=device)
        tp = torch.tensor([s['type']], device=device)
        rt = torch.tensor([s['root']], device=device)
        beat = torch.tensor([s['beat']], device=device) if with_beat else None
        pos_bin = torch.tensor([s['pos_bin']], device=device) if 'pos_bin' in s else None
        toend_bin = torch.tensor([s['toend_bin']], device=device) if 'toend_bin' in s else None
        phrase_bin = torch.tensor([s['phrase_bin']], device=device) if 'phrase_bin' in s else None
        gp, grh = model.generate(fn, tp, rt, steps=steps, temp=temp,
                                 expand=False, beat=beat,
                                 pos_bin=pos_bin, toend_bin=toend_bin,
                                 phrase_bin=phrase_bin)
        gt_p = torch.tensor(s['pc'], device=device)
        gt_r = torch.tensor(s['rhythm'], device=device)
        m = gt_p != REST
        cor_p += (gp[0] == gt_p).sum().item()
        cor_r += (grh[0] == gt_r).sum().item()
        tot += len(s['pc'])
    dt = time.time() - t0
    return cor_p / max(tot, 1), cor_r / max(tot, 1), dt / len(idxs)


# ═══════════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════════

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device} ({torch.cuda.get_device_name(0) if device == "cuda" else "CPU"})')

    samples = load_melody_windows(str(ROOT / 'data/processed/melody_llm_full_v2.json'),
                                  with_beat=args.beat,
                                  block_conds=args.block_conds,
                                  with_struct=args.struct,
                                  with_phrase=args.phrase,
                                  phrases_path=str(ROOT / 'data/processed/melody_phrases_v1.json'))
    n = len(samples)
    sp = int(n * 0.9)
    print(f'窗口数: {n} (训练 {sp} / 验证 {n - sp})'
          f'{", 含拍位条件" if args.beat else ""}'
          f'{", 块恒定和声条件" if args.block_conds else ""}'
          f'{", 结构流" if args.struct else ""}'
          f'{", RoPE" if args.rope else ""}')

    model = MelodyDiffusion(d=args.d, h=args.heads, L=args.layers, max_len=256,
                            num_funcs=len(F2ID_MELODY), num_types=len(T2ID),
                            num_beat_positions=4 if args.beat else None,
                            num_pos_bins=8 if args.struct else 0,
                            num_toend_bins=4 if args.struct else 0,
                            num_phrase_bins=6 if args.phrase else 0,
                            use_rope=args.rope).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params:,}')

    tr = DataLoader(Subset(MelodyDiffDataset(samples), range(sp)),
                    batch_size=args.batch, shuffle=True)
    vl = DataLoader(Subset(MelodyDiffDataset(samples), range(sp, n)),
                    batch_size=args.batch)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    best_gen = 0.0
    t_all = time.time()
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tl = 0.0
        n_b = 0
        for b in tr:
            fn, tp, rt, pc, rh = b[:5]
            idx = 5
            beat = None; pos_bin = None; toend_bin = None; phrase_bin = None
            if args.beat and len(b) > idx:
                beat = b[idx].to(device); idx += 1
            if args.struct and len(b) > idx + 1:
                pos_bin = b[idx].to(device); toend_bin = b[idx + 1].to(device); idx += 2
            if args.phrase and len(b) > idx:
                phrase_bin = b[idx].to(device)
            fn, tp, rt, pc, rh = [x.to(device) for x in (fn, tp, rt, pc, rh)]
            opt.zero_grad()
            loss, _, _ = model.training_loss(pc, rh, fn, tp, rt, beat=beat,
                                             pos_bin=pos_bin, toend_bin=toend_bin,
                                             phrase_bin=phrase_bin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            n_b += 1
        sched.step()

        model.eval()
        rp_acc, rr_acc = eval_recovery(model, vl, device, args.beat, args.struct)
        gp_acc, gr_acc, gen_t = eval_generation(
            model, samples[sp:], device, n=48, steps=args.steps, temp=args.temp,
            with_beat=args.beat)
        gen_score = gp_acc

        if gen_score > best_gen:
            best_gen = gen_score
            torch.save(model.state_dict(), str(ROOT / args.out))
        history.append({
            'epoch': ep + 1,
            'loss': tl / n_b,
            'recovery_pitch': rp_acc, 'recovery_rhythm': rr_acc,
            'gen_pitch': gp_acc, 'gen_rhythm': gr_acc, 'gen_sec': gen_t,
        })
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'Ep{ep+1:3d} | L:{tl/n_b:.4f} | rec p/r:{rp_acc:.3f}/{rr_acc:.3f} '
                  f'| gen p/r:{gp_acc:.3f}/{gr_acc:.3f} | {time.time()-t0:.0f}s')

    json.dump(history, open(str(ROOT / 'data/processed/diffusion_train_history.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f'总耗时 {time.time()-t_all:.0f}s | best gen pitch acc={best_gen:.3f} → {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练非自回归扩散旋律模型')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--d', type=int, default=252)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--steps', type=int, default=16, help='解码迭代步数')
    parser.add_argument('--temp', type=float, default=1.0, help='解码温度')
    parser.add_argument('--out', type=str, default='melody_diffusion_v1.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beat', action='store_true',
                        help='启用小节内拍位条件 (num_beat_positions=4)')
    parser.add_argument('--block-conds', action='store_true',
                        help='和声条件按 4 音块恒定 (与推理管线一致)')
    parser.add_argument('--struct', action='store_true',
                        help='启用结构流 (全曲位置 bin + 距结尾 bin, 数据自动推导)')
    parser.add_argument('--rope', action='store_true',
                        help='用 RoPE 相对位置编码替代可学习绝对位置 (长度外推)')
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--phrase', action='store_true',
                        help='启用乐句流 (phrase_toend bin, Ollama 标注)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train(args)
