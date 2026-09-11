"""
曲式层 —— 参考 Toward Guided Musical Form (CMJ 2024): LLM 设计曲式 + 逐句引导。

实现三件事:
    1. plan_form   : 设计乐句计划 (素材来源 reuse + 音区 register + 能量 energy),
                     LLM (本地 Ollama) 优先, 失败回退模板 (起-承-转-合 + 音区拱形);
    2. apply_motif_reuse : A-B-A' 动机复用 —— 目标乐句保留源乐句素材中
                     与当前和弦协和的音作为真值上下文, 用扩散模型补齐其余
                     (条件补全而非复制, 故必然产生变奏);
    3. apply_registers   : 按计划的音区标签整句移八度 (音区对比)。

各函数保持纯函数风格, 便于单独测试。
"""
from __future__ import annotations

import json
import random
import urllib.request

OLLAMA = 'http://localhost:11434/api/chat'
REST = 12
PITCH_MASK = 13
RHYTHM_MASK = 8


# ─────────────────────────────────────────────────────────────
# 乐句切分
# ─────────────────────────────────────────────────────────────

def phrase_spans_from_labels(labels: list[int]) -> list[tuple[int, int]]:
    """由 phrase_toend 标签 (0=句末) 得到乐句区间 [start, end] (含端点)。"""
    spans = []
    start = 0
    for i, lab in enumerate(labels):
        if lab == 0:
            spans.append((start, i))
            start = i + 1
    if start < len(labels):
        spans.append((start, len(labels) - 1))
    return spans


# ─────────────────────────────────────────────────────────────
# 曲式计划
# ─────────────────────────────────────────────────────────────

def plan_form_template(n_phrases: int, rng: random.Random) -> list[dict]:
    """模板曲式: 素材复用概率递减的 A-B-A'-C-... + 音区拱形。"""
    plan = []
    for i in range(n_phrases):
        # 复用: 第 3 句起有 60% 概率复用 2-3 句前的素材 (A-B-A' 的自然节奏)
        reuse = None
        if i >= 2 and rng.random() < 0.6:
            cands = list(range(max(0, i - 3), i - 1))
            if cands:
                reuse = rng.choice(cands)
        # 音区拱形: 前 1/3 中音区, 中段推向高音区 (climax ~70%), 结尾回落
        pos = i / max(n_phrases - 1, 1)
        if pos < 0.35:
            reg = rng.choice(['mid', 'mid', 'low'])
        elif pos < 0.75:
            reg = rng.choice(['high', 'high', 'mid'])
        else:
            reg = rng.choice(['mid', 'low', 'mid'])
        plan.append({'reuse': reuse, 'register': reg, 'energy': 2})
    return plan


def _llm_form(n_phrases: int, timeout: int = 120) -> list[dict] | None:
    prompt = (
        f'Design the musical form of a short C major piano piece with exactly '
        f'{n_phrases} phrases (each phrase is 2 measures).\n'
        f'For each phrase give:\n'
        f'- "reuse": the id of an earlier phrase whose motif this phrase should '
        f'develop (integer), or null for new material. Use motivic development: '
        f'restate the opening idea near the end (rounded form).\n'
        f'- "register": "low" | "mid" | "high" (registral contrast; put the climax '
        f'around 2/3 of the piece).\n'
        f'- "energy": 1 | 2 | 3 (dynamic intensity).\n'
        f'Reply ONLY with JSON: {{"phrases": [{{"id": 1, "reuse": null, '
        f'"register": "mid", "energy": 2}}, ...]}}'
    )
    body = {
        'model': 'qwen3.5:9b-q4_K_M',
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'think': False,
        'options': {'temperature': 0.8, 'num_predict': 512},
    }
    try:
        req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        txt = data['message']['content']
        import re
        m = re.search(r'\{.*\}', txt, re.S)
        if not m:
            return None
        obj = json.loads(m.group(0))
        items = obj.get('phrases', [])
        if len(items) != n_phrases:
            return None
        plan = []
        for it in items:
            reuse = it.get('reuse')
            if reuse is not None:
                reuse = int(reuse) - 1        # 1-based → 0-based
                if not (0 <= reuse < n_phrases) or reuse >= len(plan) + 0:
                    reuse = None
            reg = it.get('register', 'mid')
            if reg not in ('low', 'mid', 'high'):
                reg = 'mid'
            plan.append({'reuse': reuse, 'register': reg,
                         'energy': int(it.get('energy', 2))})
        # 合法性: reuse 必须指向更早且已有效的乐句
        for i, p in enumerate(plan):
            if p['reuse'] is not None and (p['reuse'] >= i or p['reuse'] < 0):
                p['reuse'] = None
        return plan
    except Exception:
        return None


def plan_form(n_phrases: int, rng: random.Random, use_llm: bool = True) -> list[dict]:
    if use_llm:
        plan = _llm_form(n_phrases)
        if plan is not None:
            return plan
    return plan_form_template(n_phrases, rng)


# ─────────────────────────────────────────────────────────────
# 动机复用 (A-B-A')
# ─────────────────────────────────────────────────────────────

def apply_motif_reuse(model, pcs, rhs, chords, spans, plan,
                      chord_tones_of_chord, keep_ratio: float = 0.5,
                      rng: random.Random | None = None, device='cuda'):
    """对 plan 中 reuse 的乐句: 保留源乐句素材中与目标和弦协和的音, 其余重生成。

    pcs/rhs: 当前全曲音符 (模型输出, 未做 polish 的音级/节奏 token);
    chords: 每和弦 dict (含 'func','chord'); 每和弦 4 音;
    spans: 乐句区间 [(s,e)]; chord_tones_of_chord(c) → [pc...]
    """
    import torch
    from model.melody_diffusion import PITCH_MASK, RHYTHM_MASK, REST as _REST
    from constants import F2ID_MELODY, T2ID
    from model.architectures import chord_name_to_info

    rng = rng or random.Random(0)
    n = len(pcs)
    pcs = list(pcs)
    rhs = list(rhs)

    # 全曲条件 (与生成时同口径)
    cf, ct, cr = [], [], []
    for c in chords:
        r, ctype = chord_name_to_info(c['chord'])
        fid = min(F2ID_MELODY.get(c['func'], 4), len(F2ID_MELODY) - 1)
        tid = min(T2ID.get(ctype, 0), len(T2ID) - 1)
        cf += [fid] * 4
        ct += [tid] * 4
        cr += [r] * 4
    cts_all = []
    for c in chords:
        tones = chord_tones_of_chord(c)
        for _ in range(4):
            cts_all.append(tones)          # 每音 → 所属和弦的音级列表

    for pi, (ps, pe) in enumerate(spans):
        if pi >= len(plan) or plan[pi]['reuse'] is None:
            continue
        src = plan[pi]['reuse']
        ss, se = spans[src]
        src_len = se - ss + 1
        tgt_len = pe - ps + 1
        if src_len < 2 or tgt_len < 2:
            continue
        # 构建 keep 掩码: 目标乐句外全部保留 (上下文), 句内保留协和素材音。
        # 动机头 (乐句前 40%) 强保留 (主题陈述), 尾部弱保留 (发展变化)。
        keep = [True] * n
        init_p = list(pcs)
        init_r = list(rhs)
        span_len = pe - ps + 1
        head_end = ps + max(1, int(span_len * 0.4))
        for i in range(ps, pe + 1):
            keep[i] = False
            # 源乐句同相对位置的音
            rel = int((i - ps) / max(tgt_len - 1, 1) * (src_len - 1))
            sp, sr = pcs[ss + rel], rhs[ss + rel]
            local_ratio = keep_ratio + 0.3 if i < head_end else max(keep_ratio - 0.25, 0.1)
            if sp < _REST and sp in cts_all[i] and rng.random() < local_ratio:
                keep[i] = True
                init_p[i] = sp
                init_r[i] = sr
        # 生成: 只重写 keep=False 的位置
        fn = torch.tensor([cf[:n]], device=device)
        tp_ = torch.tensor([ct[:n]], device=device)
        rt_ = torch.tensor([cr[:n]], device=device)
        ip = torch.tensor([init_p], device=device)
        ir = torch.tensor([init_r], device=device)
        kt = torch.tensor([keep], device=device)
        L = n
        pos = torch.tensor([[min(7, int(i / L * 8)) for i in range(L)]], device=device)
        tb = [0 if 1 - (i + 1) / L < 0.06 else (1 if 1 - (i + 1) / L < 0.15
              else (2 if 1 - (i + 1) / L < 0.4 else 3)) for i in range(L)]
        toend = torch.tensor([tb], device=device)
        # 乐句流: 目标乐句按其句内位置
        ph = []
        for (s_, e_) in spans:
            ln = e_ - s_ + 1
            for j in range(ln):
                ph.append(min(ln - 1 - j, 5))
        phb = torch.tensor([ph], device=device)
        gp, grh = model.generate_with_context(
            fn, tp_, rt_, ip, ir, kt, steps=16, temp=0.85,
            beat=None, pos_bin=pos, toend_bin=toend, phrase_bin=phb,
            seed=rng.randint(0, 10 ** 6))
        for i in range(ps, pe + 1):
            pcs[i] = int(gp[0, i].item())
            rhs[i] = int(grh[0, i].item())
    return pcs, rhs


# ─────────────────────────────────────────────────────────────
# 音区
# ─────────────────────────────────────────────────────────────

def apply_registers(midi_pitches, spans, plan, lo: int = 48, hi: int = 96):
    """按计划的音区标签整句移八度 (保持句内音程关系)。"""
    out = list(midi_pitches)
    reg_shift = {'low': -12, 'mid': 0, 'high': 12}
    for pi, (s, e) in enumerate(spans):
        if pi >= len(plan):
            continue
        shift = reg_shift.get(plan[pi].get('register', 'mid'), 0)
        if shift == 0:
            continue
        seg = [out[i] for i in range(s, min(e + 1, len(out))) if out[i] >= 0]
        if not seg:
            continue
        if min(seg) + shift < lo or max(seg) + shift > hi:
            continue
        for i in range(s, min(e + 1, len(out))):
            if out[i] >= 0:
                out[i] += shift
    return out
