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


def generate_midi(chords: list[dict], midi_pitches: list[int],
                  melody_rhythms: list[int], bpm: int,
                  output_path: str):
    """mido 直写三轨 MIDI：旋律 / 织体 / 和弦。"""
    phrases = split_phrases(chords)
    phrase_textures = {id(p): random.choice(TEXTURES) for p in phrases}

    ev_melody = []
    ev_texture = []
    ev_chord_pad = []
    tick = 0
    mel_idx = 0

    for phrase in phrases:
        tex_func = phrase_textures[id(phrase)]
        for chord in phrase:
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

            # 织体轨
            n_tex = max(1, int(dur_beats * 2))
            tex_notes = tex_func(r, ct, n=n_tex)
            for j in range(n_tex):
                t_on = tick + int(j * dur_ticks / n_tex)
                t_off = tick + int((j + 1) * dur_ticks / n_tex)
                ev_texture.append((t_on, 'on', tex_notes[j]))
                ev_texture.append((t_off, 'off', tex_notes[j]))

            # 旋律轨：保留 compose_melody 生成的节奏信息
            slot = dur_ticks / NOTES_PER_CHORD
            for j in range(NOTES_PER_CHORD):
                if mel_idx >= len(midi_pitches):
                    break

                midi = midi_pitches[mel_idx]
                rh = melody_rhythms[mel_idx] if mel_idx < len(melody_rhythms) else 2
                mel_idx += 1

                t_on = tick + int(j * slot)
                note_len = min(int(MELODY_RHYTHM.get(rh, 0.25) * TPB),
                               int(slot))
                t_off = t_on + max(note_len, 1)

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

def polish_melody(pitch_classes, rhythms, chords, chord_tones_all):
    """轻量后处理：八度拱形 + 结尾解决。scorer 已处理大部分约束。"""
    n = len(chords)
    midi_out = []; rh_out = list(rhythms)
    for idx, pc in enumerate(pitch_classes):
        c_idx = min(idx // NOTES_PER_CHORD, n - 1)
        t = idx / max(len(pitch_classes) - 1, 1)
        if t < 0.20:       oct = 0
        elif t < 0.50:     oct = 0 if t < 0.35 else 1
        elif t < 0.70:     oct = 1
        elif t < 0.88:     oct = 0
        else:              oct = -1
        if pc >= 12:
            midi_out.append(-1); continue
        midi = pc + (OCTAVE_MELODY + oct) * 12
        if c_idx >= n - 2 and idx % NOTES_PER_CHORD >= NOTES_PER_CHORD - 2:
            target = (0, 4, 7)[hash(str(idx)) % 3]
            if abs(midi % 12 - target) <= 3:
                midi = midi - (midi % 12) + target
            rh_out[idx] = min(rh_out[idx] + 2, 7)
        if idx == len(pitch_classes) - 1:
            midi = (OCTAVE_MELODY + oct) * 12; rh_out[idx] = 6
        midi_out.append(midi)
    return midi_out, rh_out


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

    # 3) MelodyGPT 生成旋律（NOTES_PER_CHORD=4 音/和弦上下文）
    print('ML 谱写旋律 ...')
    n_chords = len(chords)
    total_notes = n_chords * NOTES_PER_CHORD

    # 构建模型输入：每个和弦重复 NOTES_PER_CHORD 次
    cf_t, ct_t, cr_t = [], [], []
    for c in chords:
        r, ct = chord_name_to_info(c['chord'])
        fid = min(F2ID_MELODY.get(c['func'], 4), len(F2ID_MELODY) - 1)
        tid = min(T2ID.get(ct, 0), len(T2ID) - 1)
        cf_t.extend([fid] * NOTES_PER_CHORD)
        ct_t.extend([tid] * NOTES_PER_CHORD)
        cr_t.extend([r] * NOTES_PER_CHORD)

    cf = torch.tensor([cf_t], device=device)
    ct_tensor = torch.tensor([ct_t], device=device)
    cr = torch.tensor([cr_t], device=device)

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

    for cand in range(NUM_CANDIDATES):
        gp, grl, grh = melody_model.gen(cf, ct_tensor, cr, max_len=total_notes, temp=1.0,
                                         notes_per_chord=NOTES_PER_CHORD, scorer=melody_scorer)
        cand_pcs = [int(p) for p in gp[0].tolist()]
        cand_rhythms = [int(r) for r in grh[0].tolist()]

        # 候选旋律评分
        score = 0.0
        chord_count = 0
        step_count = 0
        repeat_count = 0
        prev_pc = None
        prev2_pc = None
        for i, pc in enumerate(cand_pcs):
            if pc >= 12:  # REST
                prev_pc = None; continue
            c_idx = i // NOTES_PER_CHORD
            cts = chord_tones_all[min(c_idx, len(chord_tones_all)-1)]
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
        score += step_count / max(n_valid, 1) * 2.0   # 级进率
        score -= repeat_count / max(n_valid, 1) * 1.5  # 重复率
        ct_ratio = chord_count / max(n_valid, 1)
        score -= abs(ct_ratio - 0.5) * 1.0  # 和弦音比例偏离 50% 就扣分
        if score > best_score:
            best_score = score
            best_pcs = cand_pcs
            best_rhythms = cand_rhythms

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
    print('完成!')

if __name__ == '__main__':
    main()
