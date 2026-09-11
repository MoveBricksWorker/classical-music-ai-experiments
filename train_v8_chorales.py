"""
v8 句子级模型训练 —— 巴赫众赞歌数据集 (完整作品 + 终止式标注)。

相比 v7 的本质变化:
    1. 数据: 完整作品 (368 首, 中位 46 音, 4.5 个乐句/首, 乐句以**终止式**结束);
    2. 窗口: 32 音 (≈ 2-3 个乐句, 模型第一次能看见完整的乐句/句子);
    3. 新增**终止式流** (乐句终止式类型: 全/半/阻碍/变格);
    4. 新增**边界预测头** (辅助任务): 判断每个位置是否乐句末
       —— "模型是否理解句子"的可操作定义与可测量指标 (边界 F1)。

用法:
    python train_v8_chorales.py --epochs 120 --batch 64
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

import argparse, json, random, time
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim

from chorale_conditions import load_chorale_windows as load_shared_windows
from constants import F2ID_MELODY, T2ID
from model.melody_diffusion import MelodyDiffusion, PITCH_MASK, RHYTHM_MASK, REST

WIN, STRIDE = 32, 6
CADENCE_ID = {'authentic': 1, 'half': 2, 'deceptive': 3, 'plagal': 4}


def load_chorale_windows(path: str, win: int = WIN, stride: int = STRIDE):
    """v8 旧口径: 条件编码见 src/chorale_conditions.py (唯一实现)。

    保留末尾的全局 shuffle + 样本级 90/10 划分是为了复现 v8 的历史数字;
    **该划分有同曲窗口泄漏 (96% 验证窗口与训练窗口同曲重叠), 新训练请用
    train_v9_chorales.py 的按曲分组划分。**
    """
    samples = load_shared_windows(path, win, stride)
    random.shuffle(samples)
    return samples


class ChoraleDataset(Dataset):
    def __init__(self, samples):
        self.s = samples

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        return (torch.tensor(s['func']), torch.tensor(s['type']), torch.tensor(s['root']),
                torch.tensor(s['pc']), torch.tensor(s['rhythm']),
                torch.tensor(s['pos_bin']), torch.tensor(s['toend_bin']),
                torch.tensor(s['phrase_bin']), torch.tensor(s['cadence']),
                torch.tensor(s['boundary']))


def batch_to_dev(b, device):
    return [x.to(device) for x in b]


@torch.no_grad()
def eval_recovery(model, loader, device, n_batches=8):
    tot = cor = 0
    for bi, b in enumerate(loader):
        if bi >= n_batches:
            break
        fn, tp, rt, pc, rh, pb, tb, phb, cad, bd = batch_to_dev(b, device)
        B, L = pc.shape
        t = torch.empty(B, device=device).uniform_(0.05, 0.95)
        mask = torch.rand(B, L, device=device) < t.unsqueeze(1)
        p_in = pc.clone(); r_in = rh.clone()
        p_in[mask] = PITCH_MASK; r_in[mask] = RHYTHM_MASK
        flag = (~mask).long() * 2 + (~mask).long()
        pl, rl, _ = model(p_in, r_in, fn, tp, rt, flag, None, pb, tb, phb, cad,
                          return_boundary=True)
        cor += (pl.argmax(-1)[mask] == pc[mask]).sum().item()
        tot += mask.sum().item()
    return cor / max(tot, 1)


@torch.no_grad()
def eval_boundary_f1(model, loader, device, threshold=0.5, n_batches=999):
    """边界预测 F1: 给定真实旋律, 判断哪些位置是乐句末。"""
    tp = fp = fn_ = 0
    for bi, b in enumerate(loader):
        if bi >= n_batches:
            break
        fn, tp_, rt, pc, rh, pb, tb, phb, cad, bd = batch_to_dev(b, device)
        flag = torch.ones_like(pc)
        # 遮蔽结构流: 边界预测必须来自旋律与和声内容 (理解句法的检验)
        phb = torch.full_like(phb, 6)
        cad = torch.zeros_like(cad)
        _, _, bl = model(pc, rh, fn, tp_, rt, flag, None, pb, tb, phb, cad,
                         return_boundary=True)
        pred = (torch.softmax(bl, -1)[..., 1] > threshold).long()
        tp += ((pred == 1) & (bd == 1)).sum().item()
        fp += ((pred == 1) & (bd == 0)).sum().item()
        fn_ += ((pred == 0) & (bd == 1)).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn_, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return f1, prec, rec


@torch.no_grad()
def eval_generation(model, samples, device, n=48, steps=16):
    """全掩码生成 (条件含真实乐句/终止式计划), 与真值对比。"""
    idxs = random.sample(range(len(samples)), min(n, len(samples)))
    cor = tot = 0
    for i in idxs:
        s = samples[i]
        fn = torch.tensor([s['func']], device=device)
        tp = torch.tensor([s['type']], device=device)
        rt = torch.tensor([s['root']], device=device)
        pb = torch.tensor([s['pos_bin']], device=device)
        tb = torch.tensor([s['toend_bin']], device=device)
        phb = torch.tensor([s['phrase_bin']], device=device)
        cad = torch.tensor([s['cadence']], device=device)
        gp, _ = model.generate(fn, tp, rt, steps=steps, temp=1.0, expand=False,
                               pos_bin=pb, toend_bin=tb, phrase_bin=phb, cadence=cad)
        gt = torch.tensor(s['pc'], device=device)
        cor += (gp[0] == gt).sum().item()
        tot += len(s['pc'])
    return cor / max(tot, 1)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')

    path = str(ROOT / 'data/processed/chorales_sentences_v1.json')
    samples = load_chorale_windows(path, args.win, args.stride)
    n = len(samples)
    sp = int(n * 0.9)
    n_bounds = sum(sum(s['boundary']) for s in samples)
    print(f'窗口数: {n} (训练 {sp} / 验证 {n - sp}) | 窗口长 {args.win}')
    print(f'边界率: {n_bounds / (n * args.win) * 100:.1f}% (每窗口约 {n_bounds / n:.1f} 个句末)')

    model = MelodyDiffusion(d=args.d, h=8, L=args.layers, max_len=256,
                            num_funcs=len(F2ID_MELODY), num_types=len(T2ID),
                            num_pos_bins=8, num_toend_bins=4, num_phrase_bins=7,
                            num_cadence_types=5, use_rope=True).to(device)
    print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

    tr = DataLoader(Subset(ChoraleDataset(samples), range(sp)), batch_size=args.batch,
                    shuffle=True)
    vl = DataLoader(Subset(ChoraleDataset(samples), range(sp, n)), batch_size=args.batch)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_score = -1
    t_all = time.time()
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tl = 0.0
        nb = 0
        for b in tr:
            fn, tp_, rt, pc, rh, pb, tb, phb, cad, bd = batch_to_dev(b, device)
            # 结构流随机丢弃: phrase_bin=6(未知)/cadence=0, 迫使边界头
            # 从旋律内容推断句末 (否则边界任务可从输入直接读出答案)
            if random.random() < args.struct_drop:
                phb = torch.full_like(phb, 6)
                cad = torch.zeros_like(cad)
            opt.zero_grad()
            loss, _, _, _ = model.training_loss(
                pc, rh, fn, tp_, rt, beat=None, pos_bin=pb, toend_bin=tb,
                phrase_bin=phb, cadence=cad, boundary=bd, boundary_weight=args.bw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); nb += 1
        sched.step()

        model.eval()
        rec = eval_recovery(model, vl, device)
        f1, prec, recall = eval_boundary_f1(model, vl, device)
        gen = eval_generation(model, samples[sp:], device)
        # 综合分数: 边界 F1 优先 (句法理解), 兼顾恢复
        score = f1 + 0.3 * gen
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), str(ROOT / args.out))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'Ep{ep+1:3d} | L:{tl/nb:.4f} | rec:{rec:.3f} | 边界F1:{f1:.3f} '
                  f'(P{prec:.2f}/R{recall:.2f}) | gen:{gen:.3f} | {time.time()-t0:.0f}s')

    print(f'总耗时 {time.time()-t_all:.0f}s | best score={best_score:.3f} → {args.out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--d', type=int, default=320)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--win', type=int, default=WIN)
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--bw', type=float, default=0.3, help='边界损失权重')
    parser.add_argument('--struct-drop', type=float, default=0.5,
                        help='结构流随机丢弃概率 (边界头抗泄漏)')
    parser.add_argument('--out', type=str, default='melody_diffusion_v8_chorales.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed)
    train(args)
