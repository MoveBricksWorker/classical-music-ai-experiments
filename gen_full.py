"""
V4 最终版 —— 乐句级织体 + ML 旋律 + 规则润色 + mido 三轨 MIDI。

改进（相比旧版）:
  - EOP 令牌正确过滤（不再泄漏到旋律生成）
  - 删除无效的 music21 Voice 死代码
  - 织体按乐句（cadence 分句）保持一致
  - 所有常量统一引用 src/constants.py
  - 魔数提取为命名常量
  - 依赖 mido 直写 MIDI（不依赖 music21 MIDI 导出）
"""
import sys, json, torch, random, os

# -- 路径初始化 --
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'processed')
sys.path.insert(0, os.path.join(ROOT, 'src'))

from constants import (
    CHORD_INTERVALS, ID2DUR, MELODY_RHYTHM,
    F2ID, ID2F, F2ID_MELODY, T2ID,
)
from model.architectures import (
    ChordGPTv4, MelodyGPT,
    build_func_chord_map, chord_name_to_info,
)
from model.melody_diffusion import MelodyDiffusion
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# ═══════════════════════════════════════════════════════════════
# 生成参数常量
# ═══════════════════════════════════════════════════════════════

OCTAVE_MELODY = 5       # 旋律 MIDI 八度偏移（中央 C = C5 = 60）
OCTAVE_CHORD = 4        # 和弦 MIDI 八度偏移
NOTES_PER_CHORD = 4     # 每个和弦的旋律音数（四分音符基底，避免三连音感）
TPB = 480               # MIDI ticks per beat
BPM_CHOICES = [72, 80, 92, 104]
CHORD_TEMP = 0.85
CHORD_RP = 1.6
TOTAL_CHORDS = 48       # 生成总和弦数

# 前缀和弦（C大调: I - IV - V7 - I，不含终止标记）
PREFIX_CHORDS = [
    {'func': 'T',  'chord': 'I',  'dur': 5, 'beat': 1, 'inv': 0, 'cad': 0},
    {'func': 'PD', 'chord': 'IV', 'dur': 3, 'beat': 2, 'inv': 0, 'cad': 0},
    {'func': 'D',  'chord': 'V7', 'dur': 3, 'beat': 3, 'inv': 0, 'cad': 0},
    {'func': 'T',  'chord': 'I',  'dur': 7, 'beat': 1, 'inv': 0, 'cad': 0},
]

# ═══════════════════════════════════════════════════════════════
# 织体发生器
# ═══════════════════════════════════════════════════════════════

def alberti_bass(root: int, chord_type: str, n: int, octave_low: int = 2) -> list[int]:
    """阿尔贝蒂低音：低-高-中-高 循环。"""
    ivs = sorted(CHORD_INTERVALS.get(chord_type, [0, 4, 7]))
    lo = root + octave_low * 12
    mid = root + (octave_low + 1) * 12 + (ivs[len(ivs) // 2] if ivs else 7)
    hi = root + (octave_low + 1) * 12 + (ivs[-1] if ivs else 12)
    return [[lo, hi, mid, hi][i % 4] for i in range(n)]


def broken_octave(root: int, chord_type: str, n: int,
                  octave_low: int = 2, octave_high: int = 3) -> list[int]:
    """破碎八度织体。"""
    ivs = sorted(CHORD_INTERVALS.get(chord_type, [0, 4, 7]))
    bass = root + octave_low * 12
    highs = [root + octave_high * 12 + iv for iv in ivs[1:]] or [bass + 12]
    result = []
    for i in range(n):
        result.append(bass if i % 2 == 0 else highs[(i // 2) % len(highs)])
    return result


def arpeggio(root: int, chord_type: str, n: int, start_octave: int = 2) -> list[int]:
    """琶音织体。"""
    ivs = sorted(CHORD_INTERVALS.get(chord_type, [0, 4, 7]))
    notes = []
    for o in range(start_octave, start_octave + 3):
        notes.extend([root + o * 12 + iv for iv in ivs])
    return [notes[i % len(notes)] for i in range(n)]


TEXTURES = [alberti_bass, broken_octave, arpeggio]

# ═══════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════

DIFFUSION_PT = os.path.join(ROOT, 'melody_diffusion_v7_phrase.pt')
DIFFUSION_KW = dict(d=288, h=8, L=8, max_len=256,
                    num_pos_bins=8, num_toend_bins=4, num_phrase_bins=6, use_rope=True)


def load_diffusion_model(device: str):
    """加载非自回归扩散旋律模型 (GETMusic 式 D3PM)。

    注意: v7 检查点里没有 `boundary_head`（边界预测头是 v8 才加的辅助头），
    旧管线也不调用它 —— 因此这里用 strict=False 并显式打印缺失/多余参数，
    避免"静默加载半个模型"。
    """
    m = MelodyDiffusion(num_funcs=len(F2ID_MELODY), num_types=len(T2ID),
                        **DIFFUSION_KW).to(device)
    sd = torch.load(DIFFUSION_PT, map_location=device, weights_only=True)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f'[load_diffusion_model] 缺失参数 (按未训练初始化): {list(missing)}')
    if unexpected:
        print(f'[load_diffusion_model] 多余参数 (忽略): {list(unexpected)}')
    m.eval()
    return m


def load_models(device: str = 'cuda'):
    """加载和弦模型和旋律模型。"""
    # -- 词表 --
    vocab_path = os.path.join(DATA, 'chord_vocab_v3.json')
    with open(vocab_path) as f:
        vocab = json.load(f)
    c2id = vocab['c2id']
    CV = len(c2id)
    id2c = {i: c for c, i in c2id.items()}

    # -- 和弦模型 (152M, d=768, 16层) --
    chord_model = ChordGPTv4(
        func_vocab=len(F2ID), chord_vocab=CV,
        dur_vocab=8, beat_vocab=5, inv_vocab=4, cad_vocab=2,
        d=768, num_layers=16,
    ).to(device)
    chord_pt = os.path.join(ROOT, 'chord_model_cloud_v4.pt')
    if not os.path.exists(chord_pt):
        raise FileNotFoundError(f"和弦模型未找到: {chord_pt}")
    chord_model.load_state_dict(
        torch.load(chord_pt, map_location=device, weights_only=True),
        strict=True,
    )
    chord_model.eval()

    # -- 旋律模型 (65M, d=504, 16层) --
    melody_model = MelodyGPT(
        d=504, h=8, L=16,
        num_funcs=len(F2ID_MELODY), num_types=len(T2ID),
        num_roles=12, num_rhythm=8, pitch_classes=14,
    ).to(device)
    melody_pt = os.path.join(ROOT, 'melody_model_cloud_v4.pt')
    if not os.path.exists(melody_pt):
        raise FileNotFoundError(f"旋律模型未找到: {melody_pt}")
    melody_model.load_state_dict(
        torch.load(melody_pt, map_location=device, weights_only=True),
        strict=True,
    )
    melody_model.eval()

    return chord_model, melody_model, c2id, id2c, CV


# ═══════════════════════════════════════════════════════════════
# 和弦生成
# ═══════════════════════════════════════════════════════════════

def generate_chords(chord_model, c2id: dict, id2c: dict, device: str = 'cuda'):
    """自回归生成和弦进行（6 维）。返回清洗后的和弦列表。"""
    pl = len(PREFIX_CHORDS)

    # 构建 prefix tensors
    pf = torch.tensor([[F2ID.get(c['func'], 7) for c in PREFIX_CHORDS]], device=device)
    pc = torch.tensor([[c2id.get(c['chord'], 3) for c in PREFIX_CHORDS]], device=device)
    pd_t = torch.tensor([[c.get('dur', 3) for c in PREFIX_CHORDS]], device=device)
    pb_t = torch.tensor([[c.get('beat', 0) for c in PREFIX_CHORDS]], device=device)
    pi_t = torch.tensor([[c.get('inv', 0) for c in PREFIX_CHORDS]], device=device)
    pk_t = torch.tensor([[c.get('cad', 0) for c in PREFIX_CHORDS]], device=device)

    # 自回归生成
    gf, gc, gd, gb, gi, gk = chord_model.gen(
        pf, pc, pd_t, pb_t, pi_t, pk_t,
        total=TOTAL_CHORDS - pl,
        temp=CHORD_TEMP, rp=CHORD_RP,
        func_chord_map=build_func_chord_map(c2id),
    )

    # 清洗：过滤所有特殊令牌（包括 EOP）
    SPECIAL_FUNC = {'<PAD>', '<SOS>', '<EOS>', 'EOP'}
    SPECIAL_CHORD = {'<PAD>', '<SOS>', '<EOS>', '<UNK>', '<EOP>'}

    chords = []
    for i in range(gf.shape[1]):
        f_id = int(gf[0, i].item())
        c_id = int(gc[0, i].item())
        f_name = ID2F.get(f_id, '?')
        c_name = id2c.get(c_id, '?')
        if f_name in SPECIAL_FUNC or c_name in SPECIAL_CHORD:
            continue
        chords.append({
            'func': f_name,
            'chord': c_name,
            'dur': int(gd[0, i].item()),
            'beat': int(gb[0, i].item()),
            'inv': int(gi[0, i].item()),
            'cad': int(gk[0, i].item()),
        })

    print(f'生成 {len(chords)} 个有效和弦')
    return chords


# ═══════════════════════════════════════════════════════════════
# 乐句分组
# ═══════════════════════════════════════════════════════════════

def split_phrases(chords: list[dict]) -> list[list[dict]]:
    """将和弦列表分割为乐句。

    由于 cadence 头不可靠（已知 bug：几乎全部输出 1），
    改用规则分句：当 D 或 PD 功能后紧跟 T 功能时，标记为终止式。
    同时限制最短乐句长度 >= 2 和弦。
    """
    if len(chords) <= 2:
        return [chords]

    # 规则标记终止式位置：D/PD → T
    cadence_indices = set()
    for i in range(1, len(chords)):
        prev_func = chords[i - 1]['func']
        curr_func = chords[i]['func']
        if prev_func in ('D', 'PD') and curr_func == 'T':
            cadence_indices.add(i)

    # 确保最后一个和弦也是乐句结尾
    cadence_indices.add(len(chords))

    phrases = []
    start = 0
    for end in sorted(cadence_indices):
        if end > start and end - start >= 0:  # 允许单和弦乐句
            phrases.append(chords[start:end])
            start = end
        elif end > start:
            start = end

    # 合并过短的乐句（< 2 和弦合并到前一个）
    merged = []
    for p in phrases:
        if len(p) < 2 and merged:
            merged[-1].extend(p)
        else:
            merged.append(p)

    # 如果全合并成一个，至少分成两段
    if len(merged) == 1 and len(merged[0]) > 8:
        mid = len(merged[0]) // 2
        merged = [merged[0][:mid], merged[0][mid:]]

    return merged


# ═══════════════════════════════════════════════════════════════
# 旋律生成（过程式：级进运动 + 和弦音落点 + 乐句呼吸）
# ═══════════════════════════════════════════════════════════════

def compose_melody(chords: list[dict]) -> tuple[list[int], list[int]]:
    """过程式旋律：整体单弧线轮廓，级进运动，仅曲末解决。

    设计原则:
      1. 固定八度（OCTAVE_MELODY），不做八度跟踪
      2. 全局轮廓: 低音区 → 爬升 → 高音区 → 下行 → 低收
      3. 方向由轮廓决定（惯性驱动）
      4. 和弦边界上自然落于和弦音
      5. 仅最后 2 和弦解决到主音
    """
    C_MAJOR = [0, 2, 4, 5, 7, 9, 11]
    n_total = len(chords) * NOTES_PER_CHORD

    notes = []
    pc = 0                     # 从 C 开始
    direction = 1               # 初始上行

    for idx in range(n_total):
        c_idx = idx // NOTES_PER_CHORD
        if c_idx >= len(chords):
            break
        chord = chords[c_idx]
        cn = chord['chord']
        r, ct = chord_name_to_info(cn)
        chord_tones = [(r + iv) % 12 for iv in CHORD_INTERVALS.get(ct, [0, 4, 7])]

        # ---- 全局轮廓 (0.0→1.0) 决定音区倾向 ----
        t = idx / max(n_total - 1, 1)
        if t < 0.15:
            bias_low = True   # 开头偏低
        elif t < 0.55:
            bias_low = False  # 爬升
        elif t < 0.75:
            bias_low = False  # 高峰
        elif t < 0.90:
            bias_low = True   # 下行
        else:
            bias_low = True   # 低收

        # 方向：倾向低音区 → 下行，倾向高音区 → 上行
        if bias_low and pc > 7:
            direction = -1
        elif not bias_low and pc < 4:
            direction = 1
        # 否则惯性保持

        # ---- 和弦边界：滑到和弦音 ----
        if idx % NOTES_PER_CHORD == 0 and pc not in chord_tones and chord_tones:
            pc = min(chord_tones, key=lambda x: min(abs(pc - x), 12 - abs(pc - x)))

        # ---- 级进移动（固定八度，不回绕） ----
        if random.random() < 0.12:
            leap = random.choice([3, 4])
        else:
            leap = random.choice([1, 2])
        step = leap * direction

        candidate = (pc + step) % 12
        if candidate not in C_MAJOR:
            for alt in [1, -1, 2, -2]:
                if ((pc + alt * direction) % 12) in C_MAJOR:
                    candidate = (pc + alt * direction) % 12
                    break

        # 防止八度回绕：如果候选音跨过 B↔C 边界，保持在同侧
        if direction > 0 and candidate < pc and pc <= 9:
            candidate = pc  # 拒绝回绕，原地踏步
        elif direction < 0 and candidate > pc and pc >= 2:
            candidate = pc  # 拒绝回绕

        pc = candidate
        midi = pc + OCTAVE_MELODY * 12  # 固定八度！

        # ---- 节奏 ----
        is_last_chord = (c_idx >= len(chords) - 2)
        is_chord_start = (idx % NOTES_PER_CHORD == 0)

        if is_last_chord and idx % NOTES_PER_CHORD >= NOTES_PER_CHORD - 2:
            rh = 6
        elif is_last_chord:
            rh = random.choice([4, 5])
        elif is_chord_start:
            rh = random.choice([3, 4])
        else:
            rh = random.choice([2, 2, 3, 4])

        notes.append((midi, rh))

    # 最后一个音强制 C（主音）
    if notes:
        notes[-1] = (OCTAVE_MELODY * 12, 6)

    return [n[0] for n in notes], [n[1] for n in notes]


# ═══════════════════════════════════════════════════════════════
# MIDI 构建
# ═══════════════════════════════════════════════════════════════

def build_midi_track(events: list, velocity: int, channel: int = 0,
                     program: int | None = None) -> MidiTrack:
    """从 (abs_tick, 'on'/'off', pitch) 事件构建 MIDI track。"""
    # 同 tick: note_off 必须先于 note_on，避免重叠音符问题
    priority = {'off': 0, 'on': 1}
    events.sort(key=lambda x: (x[0], priority.get(x[1], 0)))

    track = MidiTrack()
    if program is not None:
        track.append(Message('program_change',
                             program=program,
                             channel=channel,
                             time=0))

    last_tick = 0
    for abs_t, etype, pitch in events:
        dt = abs_t - last_tick
        last_tick = abs_t
        track.append(Message(
            'note_on' if etype == 'on' else 'note_off',
            note=pitch,
            velocity=velocity if etype == 'on' else 0,
            channel=channel,
            time=dt,
        ))
    track.append(MetaMessage('end_of_track'))
    return track


def compute_phrase_plan(chords: list[dict], notes_per_chord: int = NOTES_PER_CHORD,
                        max_phrase_chords: int = 2) -> list[int]:
    """从和声推导乐句计划 → 每音的 phrase_toend bin (0=句末换气点, 封顶 5)。

    边界来源:
      1) 和声终止式 D/PD→T (旋律应在此"收束换气");
      2) 曲末和弦;
      3) 若相邻边界间隔超过 max_phrase_chords, 中间补边界
         (训练语料的乐句约 3-15 音, 补边界使推理乐句长度落在同尺度)。
    """
    n = len(chords)
    ends = set()
    for i in range(1, n):
        if chords[i - 1].get('func') in ('D', 'PD') and chords[i].get('func') == 'T':
            ends.add(i - 1)
    ends.add(n - 1)
    # 细分过长的乐句
    ordered = sorted(ends)
    sub = set(ends)
    prev = -1
    for e in ordered:
        k = prev + max_phrase_chords
        while e - k >= max_phrase_chords:
            sub.add(k)
            k += max_phrase_chords
        prev = e
    note_ends = sorted(c * notes_per_chord + notes_per_chord - 1 for c in sub)
    labels = []
    for ci in range(n):
        for j in range(notes_per_chord):
            i = ci * notes_per_chord + j
            nxt = next((e for e in note_ends if e >= i), note_ends[-1])
            labels.append(min(nxt - i, 5))
    return labels


def compute_sentence_plan(chords: list[dict], notes_per_chord: int = NOTES_PER_CHORD,
                          max_phrase_chords: int = 4) -> tuple[list[int], list[int]]:
    """句子级规划 (两级生成的"规划器"): 从和声进行推导终止式到达点。

    与训练数据 (众赞歌) 同口径:
      - 到达点 = 终止式和弦的起始音 (全终止 I / 半终止 V / 阻碍 vi / 变格 I);
      - 半终止只在属和弦**不立即解决**时单列 (否则它只是全终止式的一半);
      - 末到达点 = 全曲最后一个音。
    返回 (phrase_bin 列表, cadence 列表), 长度 = len(chords) * notes_per_chord。
    cadence: 0=弱分割, 1=全终止, 2=半终止, 3=阻碍, 4=变格。
    """
    n = len(chords)

    def _deg(cn: str) -> str:
        cl = (cn or '').lower()
        for k in ('vii', 'vi', 'v', 'iv', 'iii', 'ii', 'i'):
            if cl.startswith(k):
                return k
        return ''

    arrivals: list[tuple[int, int]] = []
    # 1) 解决型到达点
    for i in range(1, n):
        p, c = chords[i - 1], chords[i]
        fp, fc = p.get('func'), c.get('func')
        cp, cc = _deg(p.get('chord', '')), _deg(c.get('chord', ''))
        if fp == 'D' and fc == 'T' and cc == 'vi':
            arrivals.append((i, 3))                       # 阻碍 V→vi
        elif fp == 'D' and fc == 'T':
            arrivals.append((i, 1))                       # 全终止 V→I
        elif fp == 'PD' and fc == 'T' and cp == 'iv' and cc == 'i':
            arrivals.append((i, 4))                       # 变格 IV→I
    # 2) 半终止: 到达属和弦且不立即解决到 I
    for i in range(1, n):
        if chords[i].get('func') != 'D':
            continue
        if chords[i - 1].get('func') not in ('T', 'PD'):
            continue
        nxt = chords[i + 1] if i + 1 < n else None
        if nxt is not None and nxt.get('func') == 'T' and _deg(nxt.get('chord', '')) == 'i':
            continue
        arrivals.append((i, 2))
    arrivals = sorted(set(arrivals))
    # 3) 近距离冲突: 保留强者 (全 1 > 阻碍 3 > 变格 4 > 半 2)
    strength = {1: 0, 3: 1, 4: 2, 2: 3}
    merged: list[tuple[int, int]] = []
    for idx, ct in arrivals:
        if merged and idx - merged[-1][0] < 2:
            if strength[ct] < strength[merged[-1][1]]:
                merged[-1] = (idx, ct)
            continue
        merged.append((idx, ct))
    arrivals = merged

    # 4) 音级到达点 (到达和弦首音; 末到达点 = 最后一音)
    pts: list[tuple[int, int]] = [(idx * notes_per_chord, ct) for idx, ct in arrivals]
    total = n * notes_per_chord
    last_note = total - 1
    if not pts or pts[-1][0] != last_note:
        if pts and last_note - pts[-1][0] < notes_per_chord:
            pts[-1] = (last_note, 1)                     # 并入末音, 标全终止
        else:
            pts.append((last_note, 1))
    # 5) 过长乐句补弱分割
    sub: list[tuple[int, int]] = []
    prev = -1
    for note_idx, ct in pts:
        k = prev + max_phrase_chords * notes_per_chord
        while note_idx - k >= max_phrase_chords * notes_per_chord:
            sub.append((k, 0))
            k += max_phrase_chords * notes_per_chord
        sub.append((note_idx, ct))
        prev = note_idx
    all_pts = sorted(sub)

    phrase_bin, cadence = [], []
    for i in range(total):
        nxt = next((p for p in all_pts if p[0] >= i), all_pts[-1])
        phrase_bin.append(min(nxt[0] - i, 5))
        cadence.append(nxt[1])
    return phrase_bin, cadence


def snap_to_measures(chords: list[dict], pair_prob: float = 0.3) -> list[dict]:
    """把和弦时值吸附到 4/4 小节网格: 每小节 1 个 (4拍) 或 2 个 (2+2) 和弦。

    ChordGPT 的时值是量化 ID (0.25-4 拍任意值), 和弦变化会落在非小节位置,
    旋律的等分 slot 也随之变成非节拍间距 (如 1.5 拍和弦 → 0.375 拍/音),
    与织体的八分脉冲错位 —— 听感上"旋律与和弦拍号对不上"的根源。
    """
    out = []
    i = 0
    n = len(chords)
    while i < n:
        c = dict(chords[i])
        # 乐句末判定用 D/PD→T 规则 (ChordGPT 的 cad 头已知失效, 几乎全输出 1,
        # 不能依赖; 与 split_phrases 同口径)
        next_is_t = (i + 1 < n and c.get('func') in ('D', 'PD')
                     and chords[i + 1].get('func') == 'T')
        at_phrase_end = next_is_t or (i == n - 1)
        # 成对半小节拆分: 不与乐句末/曲末和弦配对
        if (not at_phrase_end and i + 2 < n and random.random() < pair_prob):
            c['dur'] = 5                      # 2 拍
            out.append(c)
            c2 = dict(chords[i + 1]); c2['dur'] = 5
            out.append(c2)
            i += 2
        else:
            c['dur'] = 7                      # 4 拍 (整小节)
            out.append(c)
            i += 1
    return out


def generate_midi(chords: list[dict], midi_pitches: list[int],
                  melody_rhythms: list[int], bpm: int,
                  output_path: str):
    """mido 直写三轨 MIDI：旋律 / 织体 / 和弦。"""
    chords = snap_to_measures(chords)
    phrase_labels = compute_phrase_plan(chords)   # 0 = 句末换气点
    phrases = split_phrases(chords)
    phrase_textures = {id(p): random.choice(TEXTURES) for p in phrases}

    ev_melody = []
    ev_texture = []
    ev_chord_pad = []
    tick = 0
    mel_idx = 0

    for pi, phrase in enumerate(phrases):
        tex_func = phrase_textures[id(phrase)]
        for ci, chord in enumerate(phrase):
            is_final_measure = (pi == len(phrases) - 1) and (ci == len(phrase) - 1)
            cn = chord['chord']
            r, ct = chord_name_to_info(cn)
            dur_beats = ID2DUR.get(chord['dur'], 1.0)
            dur_ticks = int(dur_beats * TPB)
            ch_end = tick + dur_ticks
            ivs = sorted(CHORD_INTERVALS.get(ct, [0, 4, 7]))

            # 和弦轨：持续全时长
            for p in [r + OCTAVE_CHORD * 12 + iv for iv in ivs]:
                ev_chord_pad.append((tick, 'on', p))
                ev_chord_pad.append((ch_end, 'off', p))

            # 织体轨: 每拍 2 音 (八分脉冲), 与小节网格对齐
            n_tex = max(1, int(round(dur_beats * 2)))
            tex_notes = tex_func(r, ct, n=n_tex)
            for j in range(n_tex):
                t_on = tick + int(round(j * dur_ticks / n_tex))
                t_off = tick + int(round((j + 1) * dur_ticks / n_tex))
                ev_texture.append((t_on, 'on', tex_notes[j]))
                ev_texture.append((t_off, 'off', tex_notes[j]))

            # 旋律轨: 小节内按节奏时值比例排布 → 量化到八分网格 → 填满整小节。
            # (原实现按 和弦时值/4 等分 slot, 非节拍化时值导致与织体网格错位;
            #  纯比例归一化也会落在非网格位置, 故最终量化到 0.5 拍网格。)
            GRID = 0.5                       # 八分音符网格 (拍)
            measure_rh = [MELODY_RHYTHM.get(
                melody_rhythms[mel_idx + j] if mel_idx + j < len(melody_rhythms) else 2,
                0.5) for j in range(NOTES_PER_CHORD)]
            total_rh = sum(measure_rh)
            scale = dur_beats / total_rh if total_rh > 0 else 0
            if not (0.4 <= scale <= 2.5):    # 极端比例 → 均分兜底
                measure_rh = [1.0] * NOTES_PER_CHORD
                scale = dur_beats / NOTES_PER_CHORD
            # 累积位置 → 比例缩放 → 量化
            raw, acc = [], 0.0
            for j in range(NOTES_PER_CHORD):
                raw.append(acc * scale)
                acc += measure_rh[j]
            onsets = []
            prev = -GRID
            for pos in raw:
                p = round(pos / GRID) * GRID
                p = max(p, prev + GRID)                 # 严格递增
                p = min(p, dur_beats - GRID)            # 不越界
                onsets.append(p)
                prev = p
            if is_final_measure:
                # 终止式节奏公式: 短-短-短-长 (末音从第 3 拍持续到小节末),
                # 保证终止长音不被比例归一化压缩
                onsets = [0.0, 0.5, 1.0, max(dur_beats - 2.0, 1.0)]
            for j in range(NOTES_PER_CHORD):
                if mel_idx >= len(midi_pitches):
                    break
                midi = midi_pitches[mel_idx]
                lab = phrase_labels[mel_idx] if mel_idx < len(phrase_labels) else 5
                is_last_note = (mel_idx == len(midi_pitches) - 1)
                mel_idx += 1
                t_on = tick + int(round(onsets[j] * TPB))
                if is_final_measure and j == NOTES_PER_CHORD - 1:
                    t_off = ch_end
                else:
                    nxt = onsets[j + 1] if j + 1 < NOTES_PER_CHORD else dur_beats
                    length = max(nxt - onsets[j], GRID)
                    # 换气: 句末音缩短 ~35%, 在下一乐句前留出气口
                    if lab == 0 and not is_last_note:
                        length *= 0.65
                    t_off = tick + int(round((onsets[j] + length * 0.95) * TPB))
                if midi >= 0:
                    ev_melody.append((t_on, 'on', midi))
                    ev_melody.append((t_off, 'off', midi))

            tick = ch_end

    mid = MidiFile(ticks_per_beat=TPB)

    t0 = MidiTrack()
    t0.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    t0.append(MetaMessage('end_of_track'))
    mid.tracks.append(t0)

    # 独立 channel，全钢琴
    mid.tracks.append(build_midi_track(ev_melody,   70, channel=0, program=0))
    mid.tracks.append(build_midi_track(ev_texture,  50, channel=1, program=0))
    mid.tracks.append(build_midi_track(ev_chord_pad, 60, channel=2, program=0))

    mid.save(output_path)
    print(f'MIDI 已保存: {output_path}  (BPM:{bpm}, 3轨)')


# ═══════════════════════════════════════════════════════════════
# 旋律约束后处理（音乐理论规则，引导但不覆盖模型输出）
# ═══════════════════════════════════════════════════════════════

def assign_octaves_min_leap(pcs: list[int], base: int = 60, lo: int = 55, hi: int = 84) -> list[int]:
    """跳进最小化八度分配。

    逐音选择使 |ΔMIDI| 最小的八度 (音级线连续性优先), 并把旋律约束在
    [lo, hi] 音域内。旧的区段式八度拱形 (0→1→0→-1) 会在区段交界产生
    整八度大跳 (实测占比 24-37%), 是听感"诡异"的主要来源。
    """
    out = []
    prev = None
    for pc in pcs:
        if pc >= 12:            # REST
            out.append(-1)
            continue
        best, best_d = None, 1e9
        for octave in range(2, 7):
            midi = pc + octave * 12
            if midi < lo or midi > hi:
                continue
            d = abs(midi - base) if prev is None else abs(midi - prev)
            if d < best_d:
                best_d, best = d, midi
        out.append(best if best is not None else pc + 60)
        prev = best if best is not None else prev
    return out


def resolve_ending(midi_pitches: list[int], target_pc: int = 0, n_last: int = 4) -> list[int]:
    """结尾解决: 最后 2 音用标准终止式公式收束 (7→0 或 2→0), 末音落主音。

    比旧的"就近下移 1-2 级"更可靠: 导音/上主音级进解决到主音是古典终止
    式的核心语汇; 倒数 3/4 音轻微向主音方向靠拢。
    """
    out = list(midi_pitches)
    n = len(out)
    if n == 0:
        return out
    # 定位最后两个"实音" (末尾可能是休止, 不能只看列表末位)
    sounding = [i for i in range(n) if out[i] >= 0]
    if not sounding:
        return out
    last_i = sounding[-1]
    # 末音: 距前音最近的主音
    prev = out[sounding[-2]] if len(sounding) >= 2 else 60
    cands = [target_pc + o * 12 for o in range(2, 7)]
    out[last_i] = min(cands, key=lambda m: abs(m - prev))
    # 倒数第 2 音: 导音(B) 或 上主音(D), 取距前音近者
    if len(sounding) >= 2:
        j = sounding[-2]
        prev2 = out[sounding[-3]] if len(sounding) >= 3 else 60
        lead = target_pc + 11            # B (导音, 下方解决)
        super_ = target_pc + 2           # D (上主音)
        cands2 = [lead - 12 * k for k in range(-1, 3)] + [super_ - 12 * k for k in range(-1, 3)]
        out[j] = min(cands2, key=lambda m: abs(m - prev2))
    # 倒数 3/4 音: 轻微向主音方向靠拢 (不超过 1 个半音, 保持原有轮廓)
    for idx in range(max(0, len(sounding) - n_last), max(0, len(sounding) - 2)):
        i = sounding[idx]
        d = (out[i] % 12 - target_pc) % 12
        if 1 <= d <= 2:
            out[i] -= 1
        elif 10 <= d <= 11:
            out[i] += 1
    return out


def polish_melody(pitch_classes, rhythms, chords, chord_tones_all):
    """后处理: 跳进最小化八度分配 + 结尾解决 + 收尾长音。

    (重写自旧的区段式八度拱形版本; 旧版在区段交界产生 24-37% 的
    整八度大跳, 听感断裂。)
    """
    n = len(chords)
    midi_out = assign_octaves_min_leap(pitch_classes, base=OCTAVE_MELODY * 12,
                                       lo=OCTAVE_MELODY * 12 - 5,
                                       hi=OCTAVE_MELODY * 12 + 12)
    midi_out = resolve_ending(midi_out, target_pc=0, n_last=4)
    rh_out = list(rhythms)
    for idx in range(max(0, len(rh_out) - 4), len(rh_out)):
        rh_out[idx] = min(max(rh_out[idx], 5), 7)   # 收尾长音 (附点四分起)
    return midi_out, rh_out


def score_candidate(cand_pcs: list[int], chord_tones_all: list[list[int]]) -> float:
    """候选旋律评分: 级进率 + 和弦音比例贴近 50% + 重复惩罚。"""
    score = 0.0
    chord_count = 0
    step_count = 0
    repeat_count = 0
    prev_pc = None
    prev2_pc = None
    for i, pc in enumerate(cand_pcs):
        if pc >= 12:  # REST
            prev_pc = None
            continue
        c_idx = i // NOTES_PER_CHORD
        cts = chord_tones_all[min(c_idx, len(chord_tones_all) - 1)]
        if pc in cts:
            chord_count += 1
        if prev_pc is not None:
            d = min(abs(pc - prev_pc), 12 - abs(pc - prev_pc))
            if 1 <= d <= 2:
                step_count += 1
            if pc == prev_pc or (prev2_pc is not None and pc == prev2_pc):
                repeat_count += 1
        prev2_pc = prev_pc
        prev_pc = pc
    n_valid = sum(1 for p in cand_pcs if p < 12)
    score += step_count / max(n_valid, 1) * 2.0    # 级进率
    score -= repeat_count / max(n_valid, 1) * 1.5  # 重复率
    ct_ratio = chord_count / max(n_valid, 1)
    score -= abs(ct_ratio - 0.5) * 1.0             # 和弦音比例偏离 50% 就扣分
    return score


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    import random as _random
    _random.seed(None)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    # 1) 加载模型
    print('加载模型 ...')
    chord_model, melody_model, c2id, id2c, CV = load_models(device)

    # 2) 生成和弦进行
    print('生成和弦 ...')
    chords = generate_chords(chord_model, c2id, id2c, device)
    if len(chords) < 4:
        print('错误: 生成的和弦太少，终止。')
        return

    # 3) 旋律生成（NOTES_PER_CHORD=4 音/和弦上下文）
    use_diffusion = os.path.exists(DIFFUSION_PT)
    print(f'谱写旋律 ({"扩散非自回归" if use_diffusion else "MelodyGPT 自回归"}) ...')
    n_chords = len(chords)
    total_notes = n_chords * NOTES_PER_CHORD

    # 构建条件：每个和弦重复 NOTES_PER_CHORD 次。
    # 类型重映射到语料 LLM 标注约定 (属和弦绝大多数被标注为 M 三和弦:
    # (D,M,7) 366 次 vs (D,dom7,7) 仅 65 次) —— 避免分布外条件触发
    # 旋律模型的同音嗡鸣 (实测振荡率 62%→47%)。
    TYPE_REMAP = {'dom7': 'M', 'm7': 'm', 'M7': 'M',
                  'dim7': 'dim', 'hdim7': 'dim', 'aug': 'M'}
    cf_t, ct_t, cr_t = [], [], []
    for c in chords:
        r, ct = chord_name_to_info(c['chord'])
        ct = TYPE_REMAP.get(ct, ct)
        fid = min(F2ID_MELODY.get(c['func'], 4), len(F2ID_MELODY) - 1)
        tid = min(T2ID.get(ct, 0), len(T2ID) - 1)
        cf_t.extend([fid] * NOTES_PER_CHORD)
        ct_t.extend([tid] * NOTES_PER_CHORD)
        cr_t.extend([r] * NOTES_PER_CHORD)

    cf = torch.tensor([cf_t], device=device)
    ct_tensor = torch.tensor([ct_t], device=device)
    cr = torch.tensor([cr_t], device=device)

    # 结构流标签 (与训练同口径: 单曲从头到尾)
    L_gen = total_notes
    struct_pos = torch.tensor([[min(7, int(i / L_gen * 8)) for i in range(L_gen)]],
                              device=device)
    tb = []
    for i in range(L_gen):
        rem = 1.0 - (i + 1) / L_gen
        tb.append(0 if rem < 0.06 else (1 if rem < 0.15 else (2 if rem < 0.40 else 3)))
    struct_toend = torch.tensor([tb], device=device)
    phrase_bin = torch.tensor([compute_phrase_plan(chords)], device=device)

    # 构建 scorer：乐理评分引导采样（替代硬规则）
    chord_tones_all = []
    for c in chords:
        r, ct = chord_name_to_info(c['chord'])
        chord_tones_all.append([(r + iv) % 12 for iv in CHORD_INTERVALS.get(ct, [0, 4, 7])])

    def melody_scorer(pc, cts, prev_pc, step):
        """评分函数: 奖励级进+和弦音+轮廓，惩罚大跳+重复。"""
        score = 0.0
        # 和弦音奖励
        if pc in cts:
            score += 0.3
        # 级进奖励 (1-2 半音)
        dist = min(abs(pc - prev_pc), 12 - abs(pc - prev_pc))
        if 1 <= dist <= 2:
            score += 0.4
        elif dist >= 6:
            score -= 0.3  # 大跳惩罚
        # 重复惩罚
        if pc == prev_pc:
            score -= 0.3
        # 强拍(step%8==0)和弦音奖励
        if step % NOTES_PER_CHORD == 0:
            if pc in cts:
                score += 0.2
            else:
                score -= 0.2
        return score

    # 生成 3 个候选旋律，自动选最优
    NUM_CANDIDATES = 3
    best_score = -999
    best_pcs = None
    best_rhythms = None

    if use_diffusion:
        diff_model = load_diffusion_model(device)
        # 每个候选: 不同随机种子 → 迭代去掩码 + scorer 引导精修
        for cand in range(NUM_CANDIDATES):
            gp, grh = diff_model.generate(
                cf[:, ::NOTES_PER_CHORD], ct_tensor[:, ::NOTES_PER_CHORD],
                cr[:, ::NOTES_PER_CHORD],
                steps=16, temp=0.85, remask_steps=6, remask_ratio=0.3,
                scorer=melody_scorer, scorer_steps=4, repeat_damp=0.05,
                pos_bin=struct_pos, toend_bin=struct_toend, phrase_bin=phrase_bin,
                seed=cand * 1000 + 7, expand=True)
            cand_pcs = [int(p) for p in gp[0].tolist()]
            cand_rhythms = [int(r) for r in grh[0].tolist()]
            score = score_candidate(cand_pcs, chord_tones_all)
            if score > best_score:
                best_score, best_pcs, best_rhythms = score, cand_pcs, cand_rhythms
    else:
        for cand in range(NUM_CANDIDATES):
            gp, grl, grh = melody_model.gen(cf, ct_tensor, cr, max_len=total_notes, temp=1.0,
                                             notes_per_chord=NOTES_PER_CHORD, scorer=melody_scorer)
            cand_pcs = [int(p) for p in gp[0].tolist()]
            cand_rhythms = [int(r) for r in grh[0].tolist()]
            score = score_candidate(cand_pcs, chord_tones_all)
            if score > best_score:
                best_score, best_pcs, best_rhythms = score, cand_pcs, cand_rhythms

    print(f'  选了候选 #{list(range(NUM_CANDIDATES))} 中得分最高的 (score={best_score:.2f})')
    m_pcs = best_pcs
    m_rhythms_raw = best_rhythms

    # 3b) 轻量后处理：八度拱形 + 结尾解决
    midi_pitches, m_rhythms = polish_melody(m_pcs, m_rhythms_raw, chords, chord_tones_all)

    # 4) 导出 MIDI
    bpm = random.choice(BPM_CHOICES)
    output_path = os.path.join(DATA, 'v4_full.mid')
    generate_midi(chords, midi_pitches, m_rhythms, bpm, output_path)

    # 打印统计
    func_counts = {}
    for c in chords:
        func_counts[c['func']] = func_counts.get(c['func'], 0) + 1
    phrases = split_phrases(chords)
    print(f'统计: {len(chords)}和弦 | {len(phrases)}乐句 | 功能分布: {func_counts}')
    prog = ' - '.join(f"{c['func']}:{c['chord']}" for c in chords)
    print(f'和弦进行: {prog}')
    print('完成!')

if __name__ == '__main__':
    main()
