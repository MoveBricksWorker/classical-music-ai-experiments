"""
云服务器一键训练 — 大参数版 (d=768和弦 / d=504旋律)
GPU: 4090 24GB · PyTorch 2.5.1 · CUDA 12.4

用法:
    # 上传整个 脑洞_FINAL/ 到云服务器
    # pip install -r requirements.txt
    # python train_big.py --model both --epochs 80 --batch 32

参数量:
    和弦模型: d=768, 16层 GPT → ~60M → batch=16 需 ~6GB
    旋律模型: d=504, 16层 GPT → ~50M → batch=32 需 ~8GB
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import json, torch, time, random, argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from collections import Counter

from constants import F2ID, ID2F, F2ID_MELODY, T2ID, R2ID
from model.architectures import ChordGPT, MelodyGPT, ChordGPTv4, build_func_chord_map, chord_name_to_info


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

class ChordDataset(Dataset):
    def __init__(self, seqs, ml=128, st=16):
        self.s = []
        for s in seqs:
            L = len(s['func'])
            for i in range(0, max(1, L - ml), st):
                e = min(i + ml, L)
                if e - i < 8:
                    continue
                f = s['func'][i:e]
                c = s['chord'][i:e]
                if len(f) < ml:
                    f = f + [0] * (ml - len(f))
                    c = c + [0] * (ml - len(c))
                self.s.append((torch.tensor(f), torch.tensor(c)))
        random.shuffle(self.s)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


def load_chord_data(json_path, vocab_path):
    with open(vocab_path) as f:
        v = json.load(f)
    c2id = v['c2id']
    CV = len(c2id)
    with open(json_path) as f:
        long = json.load(f)
    seqs = []
    for s in long:
        fids = [F2ID.get(c['func'], 7) for c in s['chords']]
        cids = [c2id.get(c['chord'], 3) for c in s['chords']]
        seqs.append({'func': fids, 'chord': cids})
    return seqs, CV, c2id


def load_melody_data(json_path):
    with open(json_path) as f:
        notes = json.load(f)
    SEQ, STRIDE = 16, 4
    samples = []
    for i in range(0, len(notes) - SEQ, STRIDE):
        chunk = notes[i:i + SEQ]
        if len(chunk) < SEQ:
            continue
        samples.append({
            'func': [F2ID_MELODY.get(n.get('func', 'Other'), 4) for n in chunk],
            'type': [T2ID.get(n.get('type', 'M'), 0) for n in chunk],
            'root': [n['root'] for n in chunk],
            'pc': [n['pc'] for n in chunk],
            'role': [R2ID.get(n['role'], 10) for n in chunk],
            'rhythm': [n['rhythm'] for n in chunk],
        })
    random.shuffle(samples)
    return samples


class MelodyDataset(Dataset):
    def __init__(self, samples):
        self.s = samples

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        L = len(s['func'])
        return (torch.tensor(s['func'][:L - 1]), torch.tensor(s['type'][:L - 1]),
                torch.tensor(s['root'][:L - 1]), torch.tensor(s['pc'][:L - 1]),
                torch.tensor(s['role'][:L - 1]), torch.tensor(s['rhythm'][:L - 1]),
                torch.tensor(s['pc'][1:]), torch.tensor(s['role'][1:]),
                torch.tensor(s['rhythm'][1:]))


# ═══════════════════════════════════════════════════════════════
# 训练函数
# ═══════════════════════════════════════════════════════════════

def train_chord(epochs=80, batch_size=16, lr=2e-4, device='cuda'):
    seqs, CV, c2id = load_chord_data(
        'data/processed/long_sequences_v3.json',
        'data/processed/chord_vocab_v3.json')
    ds = ChordDataset(seqs, 128, 16)
    n = len(ds)
    sp = int(n * 0.9)
    FV = len(F2ID)
    print(f'和弦模型 (d=768, 16层, 60M): {n}样本, {sp}训练, CV={CV}')

    model = ChordGPT(func_vocab=FV, chord_vocab=CV, d=768, num_layers=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params:,}')

    tr = DataLoader(torch.utils.data.Subset(ds, range(sp)), batch_size=batch_size, shuffle=True, pin_memory=True)
    vl = DataLoader(torch.utils.data.Subset(ds, range(sp, n)), batch_size=batch_size, pin_memory=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    best = float('inf')
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tl = 0
        for fb, cb in tr:
            fb, cb = fb.to(device), cb.to(device)
            opt.zero_grad()
            fl, cl = model(fb, cb)
            L = fb.shape[1]
            lf = crit(fl[:, :-1, :].reshape(-1, FV), fb[:, 1:].reshape(-1))
            lc = crit(cl[:, :-1, :].reshape(-1, CV), cb[:, 1:].reshape(-1))
            (lf + 0.4 * lc).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += lf.item() + 0.4 * lc.item()

        model.eval()
        va = 0
        vc = 0
        vt = 0
        with torch.no_grad():
            for fb, cb in vl:
                fb, cb = fb.to(device), cb.to(device)
                fl, cl = model(fb, cb)
                L = fb.shape[1]
                va += crit(fl[:, :-1, :].reshape(-1, FV), fb[:, 1:].reshape(-1)).item()
                p = fl[:, :-1, :].argmax(-1)
                t = fb[:, 1:]
                m = t != 0
                vc += (p[m] == t[m]).sum().item()
                vt += m.sum().item()
        av = va / len(vl)
        acc = vc / max(vt, 1)
        sched.step()
        if av < best:
            best = av
            torch.save(model.state_dict(), 'chord_model_cloud_big.pt')
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  Ep{ep+1:3d}/{epochs} | L:{tl/len(tr):.4f} | V:{av:.4f} | Acc:{acc:.3f} | {time.time()-t0:.0f}s')
    print(f'Best Val:{best:.4f} → chord_model_cloud_big.pt')


# ═══════════════════════════════════════════════════════════════
# V4 增强和弦训练（6 维: func+chord+dur+beat+inv+cad）
# ═══════════════════════════════════════════════════════════════

class ChordDatasetV4(Dataset):
    """V4 数据集 —— 6 维输入。"""

    def __init__(self, seqs, ml=128, st=16):
        self.s = []
        for s in seqs:
            for i in range(0, max(1, len(s['func']) - ml), st):
                e = min(i + ml, len(s['func']))
                if e - i < 8:
                    continue
                self.s.append((
                    self._pad(s['func'][i:e], ml),
                    self._pad(s['chord'][i:e], ml),
                    self._pad(s['dur'][i:e], ml),
                    self._pad(s['beat'][i:e], ml),
                    self._pad(s['inv'][i:e], ml),
                    self._pad(s['cad'][i:e], ml),
                ))
        random.shuffle(self.s)

    @staticmethod
    def _pad(arr, ml):
        if len(arr) < ml:
            return arr + [0] * (ml - len(arr))
        return arr

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        f, c, d, b, inv, cad = self.s[i]
        return (torch.tensor(f), torch.tensor(c), torch.tensor(d),
                torch.tensor(b), torch.tensor(inv), torch.tensor(cad))


def load_chord_data_v4(json_path, vocab_path):
    """加载 V4 格式数据。"""
    with open(vocab_path) as f:
        v = json.load(f)
    c2id = v['c2id']
    CV = len(c2id)

    with open(json_path) as f:
        raw = json.load(f)

    seqs = []
    for s in raw:
        seqs.append({
            'func': [F2ID.get(c['func'], 7) for c in s['chords']],
            'chord': [c2id.get(c['chord'], 3) for c in s['chords']],
            'dur': [max(0, min(7, c.get('dur', 3))) for c in s['chords']],
            'beat': [max(0, min(4, c.get('beat', 0))) for c in s['chords']],
            'inv': [max(0, min(3, c.get('inv', 0))) for c in s['chords']],
            'cad': [max(0, min(1, c.get('cad', 0))) for c in s['chords']],
        })
    return seqs, CV, c2id


def train_chord_v4(epochs=80, batch_size=16, lr=2e-4, device='cuda'):
    """
    V4 和弦模型训练 —— 6 维预测，多任务损失。
    
    损失权重: func×1.0 + chord×0.4 + dur×0.3 + beat×0.2 + inv×0.1 + cad×0.1
    """
    seqs, CV, c2id = load_chord_data_v4(
        'data/processed/long_sequences_v4.json',
        'data/processed/chord_vocab_v3.json')
    ds = ChordDatasetV4(seqs, 128, 16)
    n = len(ds)
    sp = int(n * 0.9)
    FV, DV, BV, IV, KV = len(F2ID), 8, 5, 4, 2

    print(f'V4 和弦模型 (d=768, 16层, 60M): {n}样本, {sp}训练')
    print(f'  vocab: F={FV} C={CV} D={DV} B={BV} I={IV} K={KV}')

    model = ChordGPTv4(func_vocab=FV, chord_vocab=CV,
                       dur_vocab=DV, beat_vocab=BV,
                       inv_vocab=IV, cad_vocab=KV,
                       d=768, num_layers=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params:,}')

    tr = DataLoader(torch.utils.data.Subset(ds, range(sp)),
                    batch_size=batch_size, shuffle=True, pin_memory=True)
    vl = DataLoader(torch.utils.data.Subset(ds, range(sp, n)),
                    batch_size=batch_size, pin_memory=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    best = float('inf')
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tl = 0
        for fb, cb, db, bb, ib, kb in tr:
            fb, cb, db = fb.to(device), cb.to(device), db.to(device)
            bb, ib, kb = bb.to(device), ib.to(device), kb.to(device)
            opt.zero_grad()
            fl, cl, dl, bl, il, kl = model(fb, cb, db, bb, ib, kb)
            L = fb.shape[1]
            lf = crit(fl[:, :-1, :].reshape(-1, FV), fb[:, 1:].reshape(-1))
            lc = crit(cl[:, :-1, :].reshape(-1, CV), cb[:, 1:].reshape(-1))
            ld = crit(dl[:, :-1, :].reshape(-1, DV), db[:, 1:].reshape(-1))
            lb = crit(bl[:, :-1, :].reshape(-1, BV), bb[:, 1:].reshape(-1))
            li = crit(il[:, :-1, :].reshape(-1, IV), ib[:, 1:].reshape(-1))
            lk = crit(kl[:, :-1, :].reshape(-1, KV), kb[:, 1:].reshape(-1))
            (lf + 0.4 * lc + 0.3 * ld + 0.2 * lb + 0.1 * li + 0.1 * lk).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += (lf + 0.4 * lc + 0.3 * ld + 0.2 * lb + 0.1 * li + 0.1 * lk).item()

        model.eval()
        va, vc, vt = 0, 0, 0
        with torch.no_grad():
            for fb, cb, db, bb, ib, kb in vl:
                fb, cb, db = fb.to(device), cb.to(device), db.to(device)
                bb, ib, kb = bb.to(device), ib.to(device), kb.to(device)
                fl, cl, dl, bl, il, kl = model(fb, cb, db, bb, ib, kb)
                L = fb.shape[1]
                va += crit(fl[:, :-1, :].reshape(-1, FV), fb[:, 1:].reshape(-1)).item()
                p = fl[:, :-1, :].argmax(-1)
                t = fb[:, 1:]
                m = t != 0
                vc += (p[m] == t[m]).sum().item()
                vt += m.sum().item()
        av = va / len(vl)
        acc = vc / max(vt, 1)
        sched.step()
        if av < best:
            best = av
            torch.save(model.state_dict(), 'chord_model_cloud_v4.pt')
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  Ep{ep+1:3d}/{epochs} | L:{tl/len(tr):.4f} | V:{av:.4f} | {time.time()-t0:.0f}s')
    print(f'Best Val:{best:.4f} → chord_model_cloud_v4.pt')


def train_melody(epochs=80, batch_size=32, lr=2e-4, device='cuda'):
    samples = load_melody_data('data/processed/melody_full_v3.json')
    ds = MelodyDataset(samples)
    n = len(ds)
    sp = int(n * 0.9)
    # 新维度: PC=13, NR=11, NRT=8, NF=6, NT=8
    PC, NR, NRT = 14, 12, 8
    NF = len(F2ID_MELODY)
    NT = len(T2ID)
    print(f'旋律模型 (d=504, 16层, 50M): {n}样本, {sp}训练')

    model = MelodyGPT(d=504, h=8, L=16, num_funcs=NF, num_types=NT,
                      num_roles=NR, num_rhythm=NRT, pitch_classes=PC).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {n_params:,}')

    tr = DataLoader(torch.utils.data.Subset(ds, range(sp)), batch_size=batch_size, shuffle=True, pin_memory=True)
    vl = DataLoader(torch.utils.data.Subset(ds, range(sp, n)), batch_size=batch_size, pin_memory=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(ignore_index=0)

    best = float('inf')
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tl = 0
        for b in tr:
            fn, tp, rt, pc, rl, rh, cp, cr, crh = [x.to(device) for x in b]
            opt.zero_grad()
            pl, rll, rhll = model(fn, tp, rt, pc, rl, rh)
            L = fn.shape[1]
            lp = crit(pl[:, :-1, :].reshape(-1, PC), cp[:, 1:].reshape(-1))
            lr_ = crit(rll[:, :-1, :].reshape(-1, NR), cr[:, 1:].reshape(-1))
            lrh = crit(rhll[:, :-1, :].reshape(-1, NRT), crh[:, 1:].reshape(-1))
            (lp + 0.3 * lr_ + 0.2 * lrh).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += lp.item() + 0.3 * lr_.item() + 0.2 * lrh.item()

        model.eval()
        va = 0
        pc_c = 0
        vt = 0
        with torch.no_grad():
            for b in vl:
                fn, tp, rt, pc, rl, rh, cp, cr, crh = [x.to(device) for x in b]
                pl, rll, rhll = model(fn, tp, rt, pc, rl, rh)
                L = fn.shape[1]
                va += crit(pl[:, :-1, :].reshape(-1, PC), cp[:, 1:].reshape(-1)).item()
                pp = pl[:, :-1, :].argmax(-1)
                m = cp[:, 1:] != 0
                pc_c += (pp[m] == cp[:, 1:][m]).sum().item()
                vt += m.sum().item()
        av = va / len(vl)
        pa = pc_c / max(vt, 1)
        sched.step()
        if av < best:
            best = av
            torch.save(model.state_dict(), 'melody_model_cloud_v4.pt')
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  Ep{ep+1:3d}/{epochs} | L:{tl/len(tr):.4f} | V:{av:.4f} | Pitch:{pa:.3f} | {time.time()-t0:.0f}s')
    print(f'Best Val:{best:.4f} → melody_model_cloud_v4.pt')


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='云训练 — 大参数和弦+旋律模型')
    parser.add_argument('--model', choices=['chord', 'melody', 'both', 'chord_v4'], default='both')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device} ({torch.cuda.get_device_name(0) if device=="cuda" else "CPU"})')
    print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')
    print(f'模型规模: 和弦60M(d=768,16层) 旋律50M(d=504,16层)')
    print(f'数据: 和弦(long_sequences.json) 旋律(melody_full_v3.json, 148K音)')
    print(f'Batch: {args.batch} | Epochs: {args.epochs}')

    if args.model in ('chord', 'both'):
        train_chord(epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=device)
    if args.model in ('chord_v4',):
        train_chord_v4(epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=device)
    if args.model in ('melody', 'both'):
        train_melody(epochs=args.epochs, batch_size=max(args.batch, 32), lr=args.lr, device=device)
