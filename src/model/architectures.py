"""
统一模型架构定义 —— 项目中所有模型类的唯一来源。

模型代际:
    V1: FunctionalLSTM + ChordRealizer (model.py, ~233K params)
    V2: MiniGPT (long_model.py, ~6.4M params)
    V3: RichMusicTransformer + MelodyPredictor (rich_transformer.py, ~6.4M params)
    V4: RichGPT + MelodyTF (gen_long.py, ~6.4M / ~1M params)
    V5-local: BigGPT + MelodyGen (gen_melody_v5.py, ~25M / ~28M params)
    V5-cloud: ChordGPT + MelodyGPT (gen_cloud.py, ~60M / ~50M params)

使用方式:
    from model.architectures import ChordGPT, MelodyGPT, build_func_chord_map, chord_name_to_info
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import CHORD_INTERVALS


# ═══════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════

def build_func_chord_map(c2id: dict) -> dict[str, list[int]]:
    """从和弦词表构建 功能→和弦ID列表 的约束映射。

    用于生成时限制：D 功能只能选 V/V7/vii° 等，
    防止模型犯 i(D) 这种音乐荒谬错误。

    注意：elif 顺序很重要，viio/vii 必须在 vi 之前，
    ii 必须在 i 之前，否则前缀匹配会出错。
    """
    m: dict[str, list[int]] = {}
    SPECIAL = {'<PAD>', '<SOS>', '<EOS>', '<UNK>', '<EOP>'}
    for cn, ci in c2id.items():
        if cn in SPECIAL:
            continue
        nl = cn.lower()
        # 顺序敏感：长前缀必须在短前缀之前
        if nl.startswith('viio') or nl.startswith('vii') or nl.startswith('vo'):
            m.setdefault('D', []).append(ci)
        elif nl.startswith('vi'):
            m.setdefault('T', []).append(ci)
        elif nl.startswith('v'):
            m.setdefault('D', []).append(ci)
        elif nl.startswith('ivo') or nl.startswith('iv'):
            m.setdefault('PD', []).append(ci)
        elif nl.startswith('iiio') or nl.startswith('iii'):
            m.setdefault('Other', []).append(ci)
        elif nl.startswith('iio') or nl.startswith('ii'):
            m.setdefault('PD', []).append(ci)
        # 仅匹配 i/i°/i7/io 等（不匹配 ii/iii，它们已被上面处理）
        elif nl in ('i', 'io', 'io7', 'io6', 'i64') or (
            nl.startswith('i') and not nl.startswith('ii') and len(nl) <= 4
        ):
            m.setdefault('T', []).append(ci)
        else:
            m.setdefault('Other', []).append(ci)
    for k in m:
        m[k] = list(set(m[k]))
    return m


def chord_name_to_info(cn: str) -> tuple[int, str]:
    """和弦名 → (根音pitch_class, 和弦类型)。

    >>> chord_name_to_info('V7')
    (7, 'dom7')
    >>> chord_name_to_info('i')
    (0, 'm')
    """
    # 防御：拒绝特殊令牌
    if cn.startswith('<') or cn in ('', '?'):
        return 0, 'M'
    # 复合级数 X/Y (次属和弦等): 根音 = root(X) + root(Y) (在 C 大调级数上复合)
    DEGREE_ROOT = {'vii': 11, 'vii°': 11, 'vi': 9, 'v': 7, 'iv': 5, 'iii': 4,
                   'ii': 2, 'i': 0}
    if '/' in cn:
        head, _, tail = cn.partition('/')
        # 去掉 figure (如 V6/5, I6/4): 取字母部分
        def _degree_root(name):
            nl = name.lower().rstrip('0123456789')
            for k in ['vii', 'vi', 'v', 'iv', 'iii', 'ii', 'i']:
                if nl.startswith(k):
                    return DEGREE_ROOT[k]
            return 0
        r = (_degree_root(head) + _degree_root(tail)) % 12
    else:
        for k, v in [('viio', 11), ('vii', 11), ('vi', 9), ('v', 7),
                      ('iv', 5), ('iii', 4), ('iio', 2), ('ii', 2), ('i', 0)]:
            if cn.lower().startswith(k):
                r = v
                break
        else:
            r = 0
    nl = cn.lower()
    if 'o' in nl and '7' in nl:
        t = 'dim7'
    elif 'o' in nl:
        t = 'dim'
    elif '7' in nl:
        t = 'dom7'
    elif cn[0].isupper() or cn.startswith('V'):
        t = 'M'
    else:
        t = 'm'
    return r, t


# ═══════════════════════════════════════════════════════════════
# V1: 两层 LSTM + MLP (功能级 → 和弦级)
# ═══════════════════════════════════════════════════════════════

class FunctionalLSTM(nn.Module):
    """功能级 LSTM —— 预测下一个功能组 (T/PD/D/Sec/Other)。"""

    def __init__(self, func_vocab_size: int, embed_dim: int = 64,
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(func_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.output = nn.Linear(hidden_dim, func_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(emb, hidden)
        last_out = self.dropout(lstm_out[:, -1, :])
        return self.output(last_out), hidden

    @torch.no_grad()
    def generate(self, start_tokens, max_length=32, temperature=0.8,
                 eos_id=2, repetition_penalty=1.2, pad_id=0, sos_id=1):
        self.eval()
        device = start_tokens.device
        batch_size = start_tokens.size(0)
        generated = start_tokens.clone()
        hidden = None
        special_ids = {pad_id, sos_id, eos_id}
        for _ in range(max_length):
            context = generated[:, -32:]
            logits, hidden = self(context, hidden)
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    seen = set(generated[b].tolist()) - special_ids
                    for tid in seen:
                        logits[b, tid] = logits[b, tid] / repetition_penalty if logits[b, tid] > 0 else logits[b, tid] * repetition_penalty
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return generated


class ChordRealizer(nn.Module):
    """和弦实现器 —— 给定功能标签，预测具体和弦。"""

    def __init__(self, func_vocab_size: int, chord_vocab_size: int,
                 embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.func_embedding = nn.Embedding(func_vocab_size, embed_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, chord_vocab_size),
        )

    def forward(self, func_ids):
        return self.net(self.func_embedding(func_ids))

    def sample_chord(self, func_ids, temperature=0.5):
        self.eval()
        with torch.no_grad():
            logits = self(func_ids)
            if temperature > 0:
                return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
            return logits.argmax(dim=-1)


# ═══════════════════════════════════════════════════════════════
# V2: MiniGPT (d=256, 6层, Decoder-only)
# ═══════════════════════════════════════════════════════════════

class MiniGPT(nn.Module):
    """小型 GPT-style Transformer，用于和弦功能预测 (V2)。"""

    def __init__(self, vocab_size=8, embed_dim=256, num_heads=8,
                 num_layers=6, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        dl = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                        dim_feedforward=embed_dim * 4,
                                        dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(dl, num_layers=num_layers)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.register_buffer('causal_mask',
            torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1))

    def forward(self, x):
        B, L = x.shape
        L = min(L, self.max_seq_len)
        tok_emb = self.token_embed(x[:, :L])
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x_emb = self.dropout(tok_emb + self.pos_embed(pos))
        mask = self.causal_mask[:L, :L]
        out = self.transformer(tgt=x_emb, memory=x_emb, tgt_mask=mask, memory_mask=mask)
        return self.output(out)

    @torch.no_grad()
    def generate(self, start_token=1, max_len=256, temperature=0.85,
                 repetition_penalty=1.5, device='cuda'):
        self.eval()
        x = torch.tensor([[start_token]], device=device, dtype=torch.long)
        special = {0, 1, 2}
        for _ in range(max_len):
            ctx = x[:, -self.max_seq_len:]
            logits = self(ctx)[:, -1, :]
            for tid in set(x[0].tolist()) - special:
                logits[0, tid] = logits[0, tid] / repetition_penalty if logits[0, tid] > 0 else logits[0, tid] * repetition_penalty
            probs = F.softmax(logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            x = torch.cat([x, next_tok], dim=1)
            if next_tok.item() == 2:  # <EOS>
                break
        return x[0].cpu().tolist()


# ═══════════════════════════════════════════════════════════════
# V3: RichMusicTransformer (Encoder-Decoder) + MelodyPredictor
# ═══════════════════════════════════════════════════════════════

class RichMusicTransformer(nn.Module):
    """Encoder-Decoder Transformer，8维富特征音乐事件建模 (V3)。"""

    def __init__(self, embed_dim=256, num_heads=8, num_layers=6,
                 max_len=256, dropout=0.1,
                 func_vocab=8, chord_vocab=40, root_vocab=12,
                 inv_vocab=4, section_vocab=3, phrase_bins=16, cadence_bins=8, key_vocab=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        D = embed_dim // 8
        self.func_embed = nn.Embedding(func_vocab, D)
        self.chord_embed = nn.Embedding(chord_vocab, D)
        self.root_embed = nn.Embedding(root_vocab, D)
        self.inv_embed = nn.Embedding(inv_vocab, D)
        self.section_embed = nn.Embedding(section_vocab, D)
        self.phrase_embed = nn.Embedding(phrase_bins, D)
        self.cadence_embed = nn.Embedding(cadence_bins, D)
        self.key_embed = nn.Embedding(key_vocab, D)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        el = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                        dim_feedforward=embed_dim * 4,
                                        dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=num_layers)
        dl = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                        dim_feedforward=embed_dim * 4,
                                        dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dl, num_layers=num_layers)
        self.func_head = nn.Linear(embed_dim, func_vocab)
        self.chord_head = nn.Linear(embed_dim, chord_vocab)
        self.root_head = nn.Linear(embed_dim, root_vocab)
        self.cadence_head = nn.Linear(embed_dim, cadence_bins)
        self.inv_head = nn.Linear(embed_dim, inv_vocab)
        self.register_buffer('causal_mask',
            torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))

    def _embed_position(self, batch: dict, L: int) -> torch.Tensor:
        B = batch['func'].shape[0]
        embeds = torch.cat([
            self.func_embed(batch['func'][:, :L]),
            self.chord_embed(batch['chord'][:, :L]),
            self.root_embed(batch['root'][:, :L]),
            self.inv_embed(batch['inversion'][:, :L]),
            self.section_embed(batch['section'][:, :L]),
            self.phrase_embed(batch['phrase_pos'][:, :L]),
            self.cadence_embed(batch['cadence_dist'][:, :L]),
            self.key_embed(batch['key'][:, :L]),
        ], dim=-1)
        pos = torch.arange(L, device=embeds.device).unsqueeze(0).expand(B, -1)
        return self.dropout(embeds + self.pos_embed(pos))

    def forward(self, encoder_batch: dict, decoder_batch: dict):
        L_enc = min(encoder_batch['func'].shape[1], self.max_len)
        L_dec = min(decoder_batch['func'].shape[1], self.max_len)
        enc_emb = self._embed_position(encoder_batch, L_enc)
        memory = self.encoder(enc_emb)
        dec_emb = self._embed_position(decoder_batch, L_dec)
        tgt_mask = self.causal_mask[:L_dec, :L_dec]
        dec_out = self.decoder(tgt=dec_emb, memory=memory, tgt_mask=tgt_mask)
        return (self.func_head(dec_out), self.chord_head(dec_out),
                self.root_head(dec_out), self.cadence_head(dec_out), self.inv_head(dec_out))

    @torch.no_grad()
    def generate(self, encoder_batch: dict, max_len=128, temperature=0.85,
                 repetition_penalty=1.5, device='cuda'):
        self.eval()
        L_enc = min(encoder_batch['func'].shape[1], self.max_len)
        enc_emb = self._embed_position(encoder_batch, L_enc)
        memory = self.encoder(enc_emb)
        start_len = 2
        gen = {k: encoder_batch[k][:, -start_len:].clone()
               for k in ['func', 'chord', 'root', 'inversion', 'section',
                         'phrase_pos', 'cadence_dist', 'key']}
        special = {0, 1, 2}
        for step in range(max_len):
            L_dec = gen['func'].shape[1]
            dec_emb = self._embed_position(gen, min(L_dec, self.max_len))
            tgt_mask = self.causal_mask[:L_dec, :L_dec]
            dec_out = self.decoder(tgt=dec_emb, memory=memory, tgt_mask=tgt_mask)
            last = dec_out[:, -1:, :]
            func_logits = self.func_head(last).squeeze(1)
            for tid in set(gen['func'][0].tolist()) - special:
                func_logits[0, tid] = func_logits[0, tid] / repetition_penalty if func_logits[0, tid] > 0 else func_logits[0, tid] * repetition_penalty
            next_func = torch.multinomial(F.softmax(func_logits / temperature, dim=-1), 1)
            fid = next_func.item()
            if fid == 2:  # <EOS>
                break
            chord_logits = self.chord_head(last).squeeze(1)
            next_chord = torch.multinomial(F.softmax(chord_logits / temperature, dim=-1), 1)
            cid = min(next_chord.item(), chord_logits.shape[-1] - 1)
            root_logits = self.root_head(last).squeeze(1)
            next_root = root_logits.argmax(dim=-1, keepdim=True)
            inv_logits = self.inv_head(last).squeeze(1)
            next_inv = inv_logits.argmax(dim=-1, keepdim=True)
            cad_logits = self.cadence_head(last).squeeze(1)
            next_cad = cad_logits.argmax(dim=-1, keepdim=True)
            section_val = min(step // (max_len // 3), 2)
            phrase_val = step % 16
            key_val = encoder_batch['key'][0, 0].item()
            gen['func'] = torch.cat([gen['func'], next_func], dim=1)
            gen['chord'] = torch.cat([gen['chord'], next_chord], dim=1)
            gen['root'] = torch.cat([gen['root'], next_root], dim=1)
            gen['inversion'] = torch.cat([gen['inversion'], next_inv], dim=1)
            gen['cadence_dist'] = torch.cat([gen['cadence_dist'], next_cad], dim=1)
            for k, v in [('section', section_val), ('phrase_pos', phrase_val), ('key', key_val)]:
                gen[k] = torch.cat([gen[k], torch.tensor([[v]], device=device, dtype=torch.long)], dim=1)
        return gen


class MelodyPredictor(nn.Module):
    """和弦→旋律 MLP 预测器 (V3)。"""

    def __init__(self, embed_dim=32, hidden_dim=96,
                 num_funcs=5, num_chord_types=10, num_roles=9):
        super().__init__()
        self.func_embed = nn.Embedding(num_funcs, embed_dim)
        self.chord_type_embed = nn.Embedding(num_chord_types, embed_dim // 2)
        self.root_embed = nn.Embedding(12, embed_dim // 2)
        self.pitch_embed = nn.Embedding(12, embed_dim)
        self.role_embed = nn.Embedding(num_roles, embed_dim // 2)
        total_embed = embed_dim + embed_dim // 2 + embed_dim // 2 + embed_dim + embed_dim // 2
        self.fusion = nn.Sequential(
            nn.Linear(total_embed, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
        )
        self.pitch_head = nn.Linear(hidden_dim, 12)
        self.role_head = nn.Linear(hidden_dim, num_roles)

    def forward(self, func_id, chord_type_id, root_pc, prev_pc, prev_role):
        emb = torch.cat([
            self.func_embed(func_id),
            self.chord_type_embed(chord_type_id),
            self.root_embed(root_pc),
            self.pitch_embed(prev_pc),
            self.role_embed(prev_role),
        ], dim=-1)
        fused = self.fusion(emb)
        return self.pitch_head(fused), self.role_head(fused)

    def predict(self, func_id, chord_type_id, root_pc, prev_pc, prev_role, temperature=0.7):
        self.eval()
        with torch.no_grad():
            pitch_logits, role_logits = self.forward(
                torch.tensor([func_id]), torch.tensor([chord_type_id]),
                torch.tensor([root_pc]), torch.tensor([prev_pc]),
                torch.tensor([prev_role]))
            pitch_probs = F.softmax(pitch_logits / temperature, dim=-1)
            pc = torch.multinomial(pitch_probs, 1).item()
            role_probs = F.softmax(role_logits / temperature, dim=-1)
            role_id = torch.multinomial(role_probs, 1).item()
        return pc, role_id


# ═══════════════════════════════════════════════════════════════
# V4: RichGPT + MelodyTF (Decoder-only GPT, 6层)
# ═══════════════════════════════════════════════════════════════

class RichGPT(nn.Module):
    """Rich-feature Decoder-only GPT (V4, d=256, 6层)。"""

    def __init__(self, func_vocab=8, chord_vocab=40, root_vocab=12,
                 inv_vocab=4, section_vocab=3, phrase_bins=16,
                 cadence_bins=8, key_vocab=12, embed_dim=256,
                 num_heads=8, num_layers=6, max_len=256, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        D = embed_dim // 8
        self.func_embed = nn.Embedding(func_vocab, D)
        self.chord_embed = nn.Embedding(chord_vocab, D)
        self.root_embed = nn.Embedding(root_vocab, D)
        self.inv_embed = nn.Embedding(inv_vocab, D)
        self.section_embed = nn.Embedding(section_vocab, D)
        self.phrase_embed = nn.Embedding(phrase_bins, D)
        self.cadence_embed = nn.Embedding(cadence_bins, D)
        self.key_embed = nn.Embedding(key_vocab, D)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        dl = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                        dim_feedforward=embed_dim * 4,
                                        dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerDecoder(dl, num_layers=num_layers)
        self.func_head = nn.Linear(embed_dim, func_vocab)
        self.chord_head = nn.Linear(embed_dim, chord_vocab)
        self.register_buffer('mask',
            torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))

    def embed(self, batch, L):
        B = batch['func'].shape[0]
        e = torch.cat([
            self.func_embed(batch['func'][:, :L]),
            self.chord_embed(batch['chord'][:, :L]),
            self.root_embed(batch['root'][:, :L]),
            self.inv_embed(batch['inversion'][:, :L]),
            self.section_embed(batch['section'][:, :L]),
            self.phrase_embed(batch['phrase_pos'][:, :L]),
            self.cadence_embed(batch['cadence_dist'][:, :L]),
            self.key_embed(batch['key'][:, :L]),
        ], dim=-1)
        pos = torch.arange(L, device=e.device).unsqueeze(0).expand(B, -1)
        return self.drop(e + self.pos_embed(pos))

    def forward(self, batch):
        L = min(batch['func'].shape[1], self.max_len)
        emb = self.embed(batch, L)
        m = self.mask[:L, :L]
        out = self.transformer(tgt=emb, memory=emb, tgt_mask=m, memory_mask=m)
        return self.func_head(out), self.chord_head(out)

    @torch.no_grad()
    def generate_structured(self, prefix, total_len=48, temp=0.78, rp=2.5, top_p=0.9):
        """结构感知生成 —— 开头(建调) → 中间(展开) → 结尾(收束)。"""
        self.eval()
        gen = {k: v.clone() for k, v in prefix.items()}
        special = {0, 1, 2}
        recent = []
        b_len = int(total_len * 0.25)
        m_len = int(total_len * 0.55)
        e_start = b_len + m_len
        for step in range(total_len):
            section_val = 0 if step < b_len else (1 if step < e_start else 2)
            gen['section'][:, -1] = section_val
            L = min(gen['func'].shape[1], self.max_len)
            emb = self.embed(gen, L)
            m = self.mask[:L, :L]
            out = self.transformer(tgt=emb, memory=emb, tgt_mask=m, memory_mask=m)
            last = out[:, -1:, :]
            fl = self.func_head(last).squeeze(1)
            steps_to_end = total_len - step
            if step >= e_start and steps_to_end <= 3:
                fid = {3: 4, 2: 5, 1: 3}.get(steps_to_end, 3)  # PD→D→T
                cid = {3: 5, 2: 7, 1: 0}.get(steps_to_end, 0)  # IV→V7→I
            else:
                adj = temp * (0.7 if step < b_len else (0.6 if step >= e_start else 1.0))
                for tid in set(gen['func'][0].tolist()) - special:
                    fl[0, tid] = fl[0, tid] / rp if fl[0, tid] > 0 else fl[0, tid] * rp
                for tid in recent[-4:]:
                    fl[0, tid] /= 3.0
                if step >= e_start:
                    fl[0, 3] *= 0.8  # 结尾不要太早回T
                sl, si = torch.sort(fl, descending=True)
                cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cum > top_p] = -1e9
                sl[0, 0] = fl[0, si[0, 0]]
                nf = torch.multinomial(F.softmax(sl / adj, dim=-1), 1)
                fid = si[0, nf[0]].item()
                if fid == 2:
                    fid = 3  # EOS→T
            recent.append(fid)
            if len(recent) > 8:
                recent.pop(0)
            if not (step >= e_start and steps_to_end <= 3):
                cl = self.chord_head(last).squeeze(1)
                scl, sci = torch.sort(cl, descending=True)
                ccum = torch.cumsum(F.softmax(scl, dim=-1), dim=-1)
                scl[ccum > 0.92] = -1e9
                scl[0, 0] = cl[0, sci[0, 0]]
                nc = torch.multinomial(F.softmax(scl / temp, dim=-1), 1)
                cid = min(sci[0, nc[0]].item(), cl.shape[-1] - 1)
            gen['func'] = torch.cat([gen['func'], torch.tensor([[fid]], device=gen['func'].device)], dim=1)
            gen['chord'] = torch.cat([gen['chord'], torch.tensor([[cid]], device=gen['chord'].device)], dim=1)
            for k in ['root', 'inversion', 'section', 'phrase_pos', 'cadence_dist', 'key']:
                gen[k] = torch.cat([gen[k], gen[k][:, -1:]], dim=1)
        return gen


class MelodyTF(nn.Module):
    """旋律 Transformer (V4, d=240, 4层)。"""

    def __init__(self, d=240, h=6, L=4, num_funcs=5, num_types=7, num_roles=8):
        super().__init__()
        D = d // 5
        self.fe = nn.Embedding(num_funcs, D)
        self.te = nn.Embedding(num_types, D)
        self.re = nn.Embedding(12, D)
        self.pe = nn.Embedding(12, D)
        self.rre = nn.Embedding(num_roles, D)
        self.pos = nn.Embedding(128, d)
        dl = nn.TransformerDecoderLayer(d_model=d, nhead=h, dim_feedforward=d * 3,
                                        dropout=0.1, batch_first=True)
        self.tf = nn.TransformerDecoder(dl, L)
        self.rh = nn.Linear(d, num_roles)
        self.ph = nn.Linear(d + num_roles, 12)
        self.register_buffer('mask',
            torch.triu(torch.ones(128, 128) * float('-inf'), diagonal=1))

    @torch.no_grad()
    def gen(self, func, typ, root, sp=0, sr=0, max_len=64, temp=0.7):
        self.eval()
        dev = func.device
        gf = func.repeat_interleave(6, dim=1)[:, :max_len]
        gt = typ.repeat_interleave(6, dim=1)[:, :max_len]
        gr = root.repeat_interleave(6, dim=1)[:, :max_len]
        gp = torch.full((1, max_len), 0, dtype=torch.long, device=dev)
        grl = torch.full((1, max_len), 0, dtype=torch.long, device=dev)
        gp[0, 0] = sp
        grl[0, 0] = sr
        rh, ph = [], []
        for step in range(1, max_len):
            L = min(max_len, 128)
            e = torch.cat([self.fe(gf[:, :L]), self.te(gt[:, :L]), self.re(gr[:, :L]),
                           self.pe(gp[:, :L]), self.rre(grl[:, :L])], dim=-1)
            e = e + self.pos(torch.arange(L, device=dev).unsqueeze(0).expand(1, -1))
            m = self.mask[:L, :L]
            o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
            last = o[:, step - 1:step, :]
            rl = self.rh(last).squeeze(1)
            rp = F.softmax(rl / temp, dim=-1)
            for rid in rh[-3:]:
                rp[0, rid] /= 2.0
            rid = torch.multinomial(rp, 1).item()
            rh.append(rid)
            ro = torch.zeros(1, 1, 8, device=dev)
            ro[0, 0, rid] = 1
            pl = self.ph(torch.cat([last, ro], dim=-1)).squeeze(1)
            pp = F.softmax(pl / temp, dim=-1)
            for pid in ph[-3:]:
                pp[0, pid] /= 2.0
            pid = torch.multinomial(pp, 1).item()
            ph.append(pid)
            if len(rh) > 8:
                rh.pop(0)
                ph.pop(0)
            gp[0, step] = pid
            grl[0, step] = rid
        return gp, grl


# ═══════════════════════════════════════════════════════════════
# V5-local: BigGPT + MelodyGen (d=512/384, 12层)
# ═══════════════════════════════════════════════════════════════

class BigGPT(nn.Module):
    """V5 和弦模型 (d=512, 12层, ~25M params)。"""

    def __init__(self, c2id: dict, d: int = 512):
        super().__init__()
        self.c2id = c2id
        self.CV = len(c2id)
        self.FV = 8
        self.ml = 256
        self.f_emb = nn.Embedding(self.FV, d)
        self.c_emb = nn.Embedding(self.CV, d)
        self.pos = nn.Embedding(256, d)
        self.drop = nn.Dropout(0.1)
        dl = nn.TransformerDecoderLayer(d_model=d, nhead=8, dim_feedforward=d * 4,
                                        dropout=0.1, batch_first=True)
        self.tf = nn.TransformerDecoder(dl, 12)
        self.f_head = nn.Linear(d, self.FV)
        self.c_head = nn.Linear(d, self.CV)
        self.register_buffer('mask',
            torch.triu(torch.ones(256, 256) * float('-inf'), diagonal=1))

    def emb(self, fb, cb):
        B, L = fb.shape
        L = min(L, 256)
        return self.drop(self.f_emb(fb[:, :L]) + self.c_emb(cb[:, :L]) +
                         self.pos(torch.arange(L, device=fb.device).unsqueeze(0).expand(B, -1)))

    def forward(self, fb, cb):
        L = min(fb.shape[1], 256)
        e = self.emb(fb, cb)
        m = self.mask[:L, :L]
        o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
        return self.f_head(o), self.c_head(o)

    @torch.no_grad()
    def gen(self, pf, pc, total=24, temp=0.82, rp=1.8):
        gf = pf.clone()
        gc = pc.clone()
        special = {0, 1, 2}
        recent = []
        for step in range(total):
            L = min(gf.shape[1], 256)
            e = self.emb(gf, gc)
            m = self.mask[:L, :L]
            o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
            last = o[:, -1:, :]
            fl = self.f_head(last).squeeze(1)
            adj = temp * 0.8
            fl[0, 2] = -1e9  # <EOS>
            for tid in set(gf[0].tolist()) - special:
                fl[0, tid] = fl[0, tid] / rp if fl[0, tid] > 0 else fl[0, tid] * rp
            for tid in recent[-4:]:
                fl[0, tid] /= 3.0
            sl, si = torch.sort(fl, descending=True)
            cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
            sl[cum > 0.92] = -1e9
            sl[0, 0] = fl[0, si[0, 0]]
            nf = torch.multinomial(F.softmax(sl / adj, dim=-1), 1)
            fid = si[0, nf[0]].item()
            cl = self.c_head(last).squeeze(1)[:, :self.CV]
            scl, sci = torch.sort(cl, descending=True)
            ccum = torch.cumsum(F.softmax(scl, dim=-1), dim=-1)
            scl[ccum > 0.92] = -1e9
            scl[0, 0] = cl[0, sci[0, 0]]
            nc = torch.multinomial(F.softmax(scl / temp, dim=-1), 1)
            cid = min(sci[0, nc[0]].item(), self.CV - 1)
            recent.append(fid)
            if len(recent) > 8:
                recent.pop(0)
            gf = torch.cat([gf, torch.tensor([[fid]], device=gf.device)], dim=1)
            gc = torch.cat([gc, torch.tensor([[cid]], device=gc.device)], dim=1)
        return gf, gc


class MelodyGen(nn.Module):
    """V5 旋律模型 (d=384, 12层, ~28M params)。"""

    def __init__(self, d=384, h=8, L=12, num_funcs=7, num_types=9,
                 num_roles=12, num_rhythm=8, pitch_classes=14):
        super().__init__()
        D = d // 6
        self.fe = nn.Embedding(num_funcs, D)
        self.te = nn.Embedding(num_types, D)
        self.re = nn.Embedding(pitch_classes, D)
        self.pe = nn.Embedding(pitch_classes, D)
        self.rle = nn.Embedding(num_roles, D)
        self.rhe = nn.Embedding(num_rhythm, D)
        self.pos = nn.Embedding(128, d)
        self.drop = nn.Dropout(0.1)
        dl = nn.TransformerDecoderLayer(d_model=d, nhead=h, dim_feedforward=d * 4,
                                        dropout=0.1, batch_first=True)
        self.tf = nn.TransformerDecoder(dl, L)
        self.p_head = nn.Linear(d, pitch_classes)
        self.r_head = nn.Linear(d, num_roles)
        self.rh_head = nn.Linear(d, num_rhythm)
        self.register_buffer('mask',
            torch.triu(torch.ones(128, 128) * float('-inf'), diagonal=1))

    def forward(self, fn, tp, rt, pc, rl, rh):
        B, L = fn.shape
        L = min(L, 128)
        e = torch.cat([self.fe(fn[:, :L]), self.te(tp[:, :L]), self.re(rt[:, :L]),
                       self.pe(pc[:, :L]), self.rle(rl[:, :L]), self.rhe(rh[:, :L])], dim=-1)
        e = e + self.pos(torch.arange(L, device=fn.device).unsqueeze(0).expand(B, -1))
        m = self.mask[:L, :L]
        o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
        return self.p_head(o), self.r_head(o), self.rh_head(o)

    @torch.no_grad()
    def gen(self, func, typ, root, pc, role, rhythm, max_len=64, temp=0.75):
        self.eval()
        dev = func.device
        gp = torch.full((1, max_len), 0, dtype=torch.long, device=dev)
        gp[0, 0] = pc[0, 0].item()
        grl = torch.full((1, max_len), 0, dtype=torch.long, device=dev)
        grl[0, 0] = role[0, 0].item()
        grh = torch.full((1, max_len), 0, dtype=torch.long, device=dev)
        grh[0, 0] = rhythm[0, 0].item()
        gf = func.repeat_interleave(5, dim=1)[:, :max_len]
        gt = typ.repeat_interleave(5, dim=1)[:, :max_len]
        gr = root.repeat_interleave(5, dim=1)[:, :max_len]
        rp_hist = []
        for step in range(1, max_len):
            L = min(max_len, 128)
            e = torch.cat([self.fe(gf[:, :L]), self.te(gt[:, :L]), self.re(gr[:, :L]),
                           self.pe(gp[:, :L]), self.rle(grl[:, :L]), self.rhe(grh[:, :L])], dim=-1)
            e = e + self.pos(torch.arange(L, device=dev).unsqueeze(0).expand(1, -1))
            m = self.mask[:L, :L]
            o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
            last = o[:, step - 1:step, :]
            pl = self.p_head(last).squeeze(1)
            pp = F.softmax(pl / temp, dim=-1)
            c_root = int(gr[0, step - 1].item())
            c_type = int(gt[0, step - 1].item())
            t_map = {0: [0, 4, 7], 1: [0, 3, 7], 2: [0, 3, 6],
                     3: [0, 4, 7, 10], 4: [0, 4, 8], 5: [0, 3, 7, 10], 6: [0, 3, 6, 9]}
            cts = [(c_root + i) % 12 for i in t_map.get(c_type, [0, 4, 7])]
            for p_ in range(pp.shape[-1]):
                if p_ in cts:
                    pp[0, p_] *= 1.8
            for pid in rp_hist[-4:]:
                pp[0, pid] /= 2.0
            pp[0, :] /= pp[0, :].sum()
            pid = torch.multinomial(pp, 1).item()
            rp_hist.append(pid)
            if len(rp_hist) > 8:
                rp_hist.pop(0)
            rl_out = self.r_head(last).squeeze(1)
            rid = torch.multinomial(F.softmax(rl_out / temp, dim=-1), 1).item()
            rh_out = self.rh_head(last).squeeze(1)
            rhid = torch.multinomial(F.softmax(rh_out / temp, dim=-1), 1).item()
            gp[0, step] = pid
            grl[0, step] = rid
            grh[0, step] = rhid
        return gp, grl, grh


# ═══════════════════════════════════════════════════════════════
# V5-cloud: ChordGPT + MelodyGPT (d=768/504, 16层, 60M/50M)
# ═══════════════════════════════════════════════════════════════

class ChordGPT(nn.Module):
    """云训练和弦模型 (d=768, 16层, ~60M params, 93.5%准确率)。"""

    def __init__(self, func_vocab: int = 9, chord_vocab: int = 63, d: int = 768,
                 num_layers: int = 16, max_len: int = 256):
        super().__init__()
        self.ml = max_len
        self.f_emb = nn.Embedding(func_vocab, d)
        self.c_emb = nn.Embedding(chord_vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(0.1)
        dl = nn.TransformerDecoderLayer(d_model=d, nhead=8, dim_feedforward=d * 4,
                                        dropout=0.1, batch_first=True)
        self.tf = nn.TransformerDecoder(dl, num_layers)
        self.f_head = nn.Linear(d, func_vocab)
        self.c_head = nn.Linear(d, chord_vocab)
        self.register_buffer('mask',
            torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))

    def emb(self, fb, cb):
        B, L = fb.shape
        L = min(L, self.ml)
        return self.drop(self.f_emb(fb[:, :L]) + self.c_emb(cb[:, :L]) +
                         self.pos(torch.arange(L, device=fb.device).unsqueeze(0).expand(B, -1)))

    def forward(self, fb, cb):
        L = min(fb.shape[1], self.ml)
        e = self.emb(fb, cb)
        m = self.mask[:L, :L]
        o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
        return self.f_head(o), self.c_head(o)

    @torch.no_grad()
    def gen(self, pf, pc, total=48, temp=0.80, rp=1.6, func_chord_map=None):
        gf = pf.clone()
        gc = pc.clone()
        special = {0, 1, 2}
        recent = []
        recent_chords = []
        bl = int(total * 0.25)
        es = int(total * 0.80)
        for step in range(total):
            sv = 0 if step < bl else (1 if step < es else 2)
            L = min(gf.shape[1], self.ml)
            e = self.emb(gf, gc)
            m = self.mask[:L, :L]
            o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
            last = o[:, -1:, :]
            fl = self.f_head(last).squeeze(1)
            ste = total - step
            if step >= es and ste <= 3:
                fid = {3: 4, 2: 5, 1: 3}.get(ste, 3)
                cid = {3: 5, 2: 7, 1: 0}.get(ste, 0)
            else:
                adj = temp * (1.0 if step < bl else (0.8 if step >= es else 1.0))
                fl[0, 2] = -1e9  # <EOS>
                for tid in set(gf[0].tolist()) - special:
                    fl[0, tid] = fl[0, tid] / rp if fl[0, tid] > 0 else fl[0, tid] * rp
                for tid in recent[-6:]:
                    fl[0, tid] /= 4.0
                if step >= es:
                    fl[0, 3] *= 0.8  # T
                sl, si = torch.sort(fl, descending=True)
                cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cum > 0.92] = -1e9
                sl[0, 0] = fl[0, si[0, 0]]
                nf = torch.multinomial(F.softmax(sl / adj, dim=-1), 1)
                fid = si[0, nf[0]].item()
                if func_chord_map:
                    from collections import Counter
                    fname = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: 'T',
                             4: 'PD', 5: 'D', 6: 'Sec', 7: 'Other'}.get(fid, 'Other')
                    valid = func_chord_map.get(fname, list(range(self.c_head.out_features)))
                else:
                    valid = list(range(self.c_head.out_features))
                cl = self.c_head(last).squeeze(1)
                mask_t = torch.ones(cl.shape[-1], device=gf.device) * -1e9
                for vc in valid:
                    if vc < cl.shape[-1]:
                        mask_t[vc] = 0
                cl = cl + mask_t
                scl, sci = torch.sort(cl, descending=True)
                ccum = torch.cumsum(F.softmax(scl, dim=-1), dim=-1)
                scl[ccum > 0.92] = -1e9
                scl[0, 0] = cl[0, sci[0, 0]]
                # 和弦级重复惩罚
                for cid_hist in recent_chords[-6:]:
                    if cid_hist < len(scl[0]):
                        scl[0, cid_hist] /= 2.0
                nc = torch.multinomial(F.softmax(scl / temp, dim=-1), 1)
                cid = min(sci[0, nc[0]].item(), cl.shape[-1] - 1)
            recent.append(fid)
            if len(recent) > 12:
                recent.pop(0)
            recent_chords.append(cid)
            if len(recent_chords) > 12:
                recent_chords.pop(0)
            gf = torch.cat([gf, torch.tensor([[fid]], device=gf.device)], dim=1)
            gc = torch.cat([gc, torch.tensor([[cid]], device=gc.device)], dim=1)
        return gf, gc


class ChordGPTv4(nn.Module):
    """
    V4 和弦模型 —— 6 维输入（func+chord+dur+beat+inv+cad）。

    在 ChordGPT 基础上新增时值/拍位/转位/终止式 4 个维度的
    embedding 和预测头。Transformer 主体不变（16层, d=768）。
    """

    def __init__(self, func_vocab: int = 8, chord_vocab: int = 62,
                 dur_vocab: int = 8, beat_vocab: int = 5,
                 inv_vocab: int = 4, cad_vocab: int = 2,
                 d: int = 768, num_layers: int = 16, max_len: int = 256):
        super().__init__()
        self.ml = max_len
        D = d
        self.f_emb = nn.Embedding(func_vocab, D)
        self.c_emb = nn.Embedding(chord_vocab, D)
        self.dur_emb = nn.Embedding(dur_vocab, D)
        self.beat_emb = nn.Embedding(beat_vocab, D)
        self.inv_emb = nn.Embedding(inv_vocab, D)
        self.cad_emb = nn.Embedding(cad_vocab, D)
        self.pos = nn.Embedding(max_len, D)
        self.drop = nn.Dropout(0.1)

        dl = nn.TransformerDecoderLayer(d_model=D, nhead=8, dim_feedforward=D * 4,
                                        dropout=0.1, batch_first=True)
        self.tf = nn.TransformerDecoder(dl, num_layers)

        self.f_head = nn.Linear(D, func_vocab)
        self.c_head = nn.Linear(D, chord_vocab)
        self.dur_head = nn.Linear(D, dur_vocab)
        self.beat_head = nn.Linear(D, beat_vocab)
        self.inv_head = nn.Linear(D, inv_vocab)
        self.cad_head = nn.Linear(D, cad_vocab)

        self.register_buffer('causal_mask',
            torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1))

    def emb(self, fb, cb, db, bb, ib, kb):
        B, L = fb.shape
        L = min(L, self.ml)
        pos_idx = torch.arange(L, device=fb.device).unsqueeze(0).expand(B, -1)
        return self.drop(
            self.f_emb(fb[:, :L]) + self.c_emb(cb[:, :L]) +
            self.dur_emb(db[:, :L]) + self.beat_emb(bb[:, :L]) +
            self.inv_emb(ib[:, :L]) + self.cad_emb(kb[:, :L]) +
            self.pos(pos_idx)
        )

    def forward(self, fb, cb, db, bb, ib, kb):
        """
        Args:
            fb, cb, db, bb, ib, kb: [B, L] tensors for func/chord/dur/beat/inv/cad

        Returns:
            (fl, cl, dl, bl, il, kl): 6 组 logits [B, L, vocab]
        """
        L = min(fb.shape[1], self.ml)
        e = self.emb(fb, cb, db, bb, ib, kb)
        m = self.causal_mask[:L, :L]
        o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
        return (self.f_head(o), self.c_head(o), self.dur_head(o),
                self.beat_head(o), self.inv_head(o), self.cad_head(o))

    @torch.no_grad()
    def gen(self, pf, pc, pd, pb, pi, pk, total=48, temp=0.80, rp=1.6,
            func_chord_map=None):
        """
        自回归生成 —— 同时生成 func/chord/dur/beat/inv/cad。

        Args:
            pf, pc, pd, pb, pi, pk: prefix tensors [1, prefix_len]
            total: 生成步数
            temp: 温度
            rp: 重复惩罚
            func_chord_map: func→[chord_ids] 约束映射

        Returns:
            (gf, gc, gd, gb, gi, gk): 完整序列 tensors
        """
        FV = self.f_head.out_features
        CV = self.c_head.out_features
        DV = self.dur_head.out_features
        BV = self.beat_head.out_features
        IV = self.inv_head.out_features
        KV = self.cad_head.out_features

        gf, gc = pf.clone(), pc.clone()
        gd, gb = pd.clone(), pb.clone()
        gi, gk = pi.clone(), pk.clone()

        recent_f = []
        recent_c = []
        recent_chord_names = []  # 最近的和弦 ID 防止重复

        for step in range(total):
            L = gf.shape[1]
            ctx = min(L, self.ml)
            e = self.emb(gf[:, -ctx:], gc[:, -ctx:], gd[:, -ctx:],
                         gb[:, -ctx:], gi[:, -ctx:], gk[:, -ctx:])
            m = self.causal_mask[:ctx, :ctx]
            o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)

            fl = self.f_head(o)[:, -1, :] / max(temp, 0.1)
            cl = self.c_head(o)[:, -1, :] / max(temp, 0.1)
            dl = self.dur_head(o)[:, -1, :] / max(temp, 0.1)
            bl = self.beat_head(o)[:, -1, :] / max(temp, 0.1)
            il = self.inv_head(o)[:, -1, :] / max(temp, 0.1)
            kl = self.cad_head(o)[:, -1, :] / max(temp, 0.1)

            # 重复惩罚
            for rid in set(recent_f):
                fl[0, rid] /= rp * 1.5
            for rid in set(recent_chord_names):
                cl[0, rid] /= rp

            # 近期惩罚：最后 6 个 function 被强力压制
            for rid in recent_f[-6:]:
                fl[0, rid] /= 4.0

            # 特殊 token 屏蔽 (PAD/SOS/EOS/EOP)
            for tid in [0, 1, 2]:
                if tid < FV: fl[0, tid] = -1e9
                if tid < CV: cl[0, tid] = -1e9
            # EOP token 也屏蔽（id=8 in F2ID, id=4 in c2id）
            if 8 < FV: fl[0, 8] = -1e9
            if 4 < CV: cl[0, 4] = -1e9

            # Nucleus (top-p=0.9) sampling for func
            sorted_fl, sorted_idx = torch.sort(fl, descending=True)
            cumsum_fl = torch.cumsum(F.softmax(sorted_fl, dim=-1), dim=-1)
            mask_fl = cumsum_fl > 0.9
            mask_fl[..., 0] = False  # 始终保留 top-1
            for j in range(1, mask_fl.shape[1]):
                if mask_fl[0, j]:
                    fl[0, sorted_idx[0, j]] -= 1e9

            probs_f = F.softmax(fl, dim=-1)
            fid = torch.multinomial(probs_f, 1).item()

            # 功能→和弦约束
            if func_chord_map:
                fname = {3: 'T', 4: 'PD', 5: 'D', 6: 'Sec', 7: 'Other'}.get(fid)
                valid = func_chord_map.get(fname, list(range(CV)))
                mask_c = torch.full((1, CV), -1e9, device=cl.device)
                for v in valid:
                    if v < CV:
                        mask_c[0, v] = 0
                cl = cl + mask_c

            # Nucleus (top-p=0.92) for chord
            sorted_cl, sorted_ci = torch.sort(cl, descending=True)
            cumsum_cl = torch.cumsum(F.softmax(sorted_cl, dim=-1), dim=-1)
            mask_cl = cumsum_cl > 0.92
            mask_cl[..., 0] = False
            for j in range(1, mask_cl.shape[1]):
                if mask_cl[0, j]:
                    cl[0, sorted_ci[0, j]] -= 1e9

            probs_c = F.softmax(cl, dim=-1)
            cid = torch.multinomial(probs_c, 1).item()

            # 采样辅助维度
            did = torch.multinomial(F.softmax(dl, dim=-1), 1).item()
            bid = torch.multinomial(F.softmax(bl, dim=-1), 1).item()
            iid = torch.multinomial(F.softmax(il, dim=-1), 1).item()
            kid = torch.multinomial(F.softmax(kl, dim=-1), 1).item()

            # 更新历史
            recent_f.append(fid)
            recent_c.append(cid)
            recent_chord_names.append(cid)
            if len(recent_f) > 12:
                recent_f.pop(0)
            if len(recent_c) > 24:
                recent_c.pop(0)
            if len(recent_chord_names) > 12:
                recent_chord_names.pop(0)

            gf = torch.cat([gf, torch.tensor([[fid]], device=gf.device)], dim=1)
            gc = torch.cat([gc, torch.tensor([[cid]], device=gc.device)], dim=1)
            gd = torch.cat([gd, torch.tensor([[did]], device=gd.device)], dim=1)
            gb = torch.cat([gb, torch.tensor([[bid]], device=gb.device)], dim=1)
            gi = torch.cat([gi, torch.tensor([[iid]], device=gi.device)], dim=1)
            gk = torch.cat([gk, torch.tensor([[kid]], device=gk.device)], dim=1)

        return gf, gc, gd, gb, gi, gk


class MelodyGPT(nn.Module):
    """云训练旋律模型 (d=504, 16层, ~50M params, 84.0%准确率)。"""

    def __init__(self, d=504, h=8, L=16, num_funcs=7, num_types=9,
                 num_roles=12, num_rhythm=8, pitch_classes=14):
        super().__init__()
        D = d // 6
        self.fe = nn.Embedding(num_funcs, D)
        self.te = nn.Embedding(num_types, D)
        self.re = nn.Embedding(pitch_classes, D)
        self.pe = nn.Embedding(pitch_classes, D)
        self.rle = nn.Embedding(num_roles, D)
        self.rhe = nn.Embedding(num_rhythm, D)
        self.pos = nn.Embedding(128, d)
        self.drop = nn.Dropout(0.1)
        dl = nn.TransformerDecoderLayer(d_model=d, nhead=h, dim_feedforward=d * 4,
                                        dropout=0.1, batch_first=True)
        self.tf = nn.TransformerDecoder(dl, L)
        self.p_head = nn.Linear(d, pitch_classes)
        self.r_head = nn.Linear(d, num_roles)
        self.rh_head = nn.Linear(d, num_rhythm)
        self.register_buffer('mask',
            torch.triu(torch.ones(128, 128) * float('-inf'), diagonal=1))

    def forward(self, fn, tp, rt, pc, rl, rh):
        B, L = fn.shape
        L = min(L, 128)
        e = torch.cat([self.fe(fn[:, :L]), self.te(tp[:, :L]), self.re(rt[:, :L]),
                       self.pe(pc[:, :L]), self.rle(rl[:, :L]), self.rhe(rh[:, :L])], dim=-1)
        e = e + self.pos(torch.arange(L, device=fn.device).unsqueeze(0).expand(B, -1))
        m = self.mask[:L, :L]
        o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
        return self.p_head(o), self.r_head(o), self.rh_head(o)

    @torch.no_grad()
    def gen(self, func, typ, root, max_len=96, temp=0.75, notes_per_chord=6,
            scorer=None):
            """自回归生成旋律。scorer(pc, chord_tones, prev_pc, step_idx) → bonus。"""
            self.eval()
            dev = func.device
            gf = func.repeat_interleave(notes_per_chord, dim=1)[:, :max_len]
            gt = typ.repeat_interleave(notes_per_chord, dim=1)[:, :max_len]
            gr = root.repeat_interleave(notes_per_chord, dim=1)[:, :max_len]
            gp = torch.full((1, max_len), 0, dtype=torch.long, device=dev)
            gp[0, 0] = 0
            grl = torch.full((1, max_len), 0, dtype=torch.long, device=dev)
            grh = torch.full((1, max_len), 1, dtype=torch.long, device=dev)
            rp_hist = []
            prev_pc = 0
            for step in range(1, max_len):
                # 滑动窗口: 只取最后128步
                ctx_len = min(step, 128)
                start = step - ctx_len
                e = torch.cat([self.fe(gf[:, start:step]), self.te(gt[:, start:step]),
                               self.re(gr[:, start:step]), self.pe(gp[:, start:step]),
                               self.rle(grl[:, start:step]), self.rhe(grh[:, start:step])], dim=-1)
                e = e + self.pos(torch.arange(ctx_len, device=dev).unsqueeze(0).expand(1, -1))
                m = self.mask[:ctx_len, :ctx_len]
                o = self.tf(tgt=e, memory=e, tgt_mask=m, memory_mask=m)
                last = o[:, -1:, :]
                pl = self.p_head(last).squeeze(1)
                pp = F.softmax(pl / temp, dim=-1)
                c_root = int(gr[0, step - 1].item())
                c_type = int(gt[0, step - 1].item())
                t_map = {0: [0, 4, 7], 1: [0, 3, 7], 2: [0, 3, 6],
                         3: [0, 4, 7, 10], 4: [0, 4, 8], 5: [0, 3, 7, 10], 6: [0, 3, 6, 9]}
                cts = [(c_root + i) % 12 for i in t_map.get(c_type, [0, 4, 7])]
                for p_ in range(pp.shape[-1]):
                    if p_ in cts:
                        pp[0, p_] *= 1.15  # 轻量和弦音引导（不锁死）
                for pid in rp_hist[-4:]:
                    pp[0, pid] /= 1.2  # 轻量重复惩罚（允许自然反复）
                pp[0, :] /= pp[0, :].sum()
                # 反振荡: 最近4音只有2个pitch且都是和弦音 → 强制排除它们
                blocked = set()
                if len(rp_hist) >= 4:
                    last4 = rp_hist[-4:]
                    if len(set(last4)) <= 2 and all(p in cts for p in last4):
                        blocked = set(last4)
                        for b in blocked:
                            pp[0, b] *= 0.01  # 强力抑制振荡音
                        pp[0, :] /= pp[0, :].sum()
                # scorer 介入: top-20 中按乐理评分重排
                if scorer is not None:
                    topk = min(20, pp.shape[-1])
                    vals, idxs = torch.topk(pp[0], topk)
                    scores = torch.zeros(topk, device=dev)
                    for k in range(topk):
                        scores[k] = scorer(int(idxs[k].item()),
                                           cts, prev_pc, step)
                    # 模型概率 × scorer bonus
                    reweighted = vals * torch.exp(scores * 0.5)
                    reweighted /= reweighted.sum()
                    pick = torch.multinomial(reweighted, 1)
                    pid = int(idxs[pick[0]].item())
                else:
                    pid = torch.multinomial(pp, 1).item()
                prev_pc = pid
                rp_hist.append(pid)
                if len(rp_hist) > 8:
                    rp_hist.pop(0)
                rl_out = self.r_head(last).squeeze(1)
                rid = torch.multinomial(F.softmax(rl_out / temp, dim=-1), 1).item()
                rh_out = self.rh_head(last).squeeze(1)
                rhid = torch.multinomial(F.softmax(rh_out / temp, dim=-1), 1).item()
                gp[0, step] = pid
                grl[0, step] = rid
                grh[0, step] = rhid
            return gp, grl, grh
