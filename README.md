# classical-music-ai-experiments
Can AI understand classical music? This project explores the gap between statistical music generation and musical structure through experiments with LLMs, symbolic music analysis, and neural generation.
# 🎼 古典和声智能生成系统

> A neuro-symbolic classical music generation system combining functional harmony modeling, neural melody generation, and music-theory-guided realization.

## 概述

基于 Transformer 的古典音乐生成系统。给定起始和弦，自动生成完整的功能性和声进行、旋律和钢琴织体，输出三轨 MIDI 文件。

**核心理念**：让模型理解"为什么这个和弦在这里"的语法，而非"这个和弦之后大概率是什么"的统计。

## 架构

```
ChordGPTv4 (152M)           MelodyGPT (65M)
6维自回归生成              和声条件旋律生成
func+chord+dur+beat         scorer引导采样
  +inv+cad                    + 反振荡拦截
      ↓                          ↓
  48个和弦进行    →    旋律音高 + 节奏
      ↓                          ↓
      └────────┬─────────────────┘
               ↓
      织体生成 (阿尔贝蒂/破碎八度/琶音)
               ↓
        三轨 MIDI 输出
    (旋律 ch0 / 织体 ch1 / 和弦 ch2)
```

## 快速开始

> ⚠️ **入口说明（2026-09-11 修订）**：下面这一节是**原版 ChordGPTv4 管线**
> 的用法（需要 579MB 和弦权重）。当前推荐入口是**句子级生成器**
> `python gen_sentence.py --mode major --seed 5`，见文末「迭代 7」。
> 各组件"哪些是模型生成、哪些来自语料检索、哪些是规则修正"见迭代 7 的表格。

### 安装

```bash
pip install -r requirements.txt
```

### 下载模型权重

原版两个权重在 Releases（`model` release）：
`https://github.com/MoveBricksWorker/classical-music-ai-experiments/releases`
下载以下文件放到项目根目录：

- `chord_model_cloud_v4.pt` (579 MB)
- `melody_model_cloud_v4.pt` (249 MB)

### 生成

```bash
python gen_full.py
```

输出：`data/processed/v4_full.mid`

### 训练

```bash
# V4 和弦模型 (6维)
python train_big.py --model chord_v4 --epochs 80 --batch 16

# 旋律模型
python train_big.py --model melody --epochs 80 --batch 32
```

## 项目结构

```
├── gen_full.py              # 主生成管线
├── train_big.py             # 云端训练脚本
├── requirements.txt
├── src/
│   ├── constants.py         # 全局常量 (和弦音程/ID映射)
│   └── model/
│       └── architectures.py # 所有模型类定义 (V1-V5)
└── data/
    └── processed/
        ├── chord_vocab_v3.json        # 63类和弦词表
        ├── long_sequences_v4.json     # 461首/31651和弦 (V4 6维)
        └── melody_llm_full_v2.json    # 13592旋律音 (LLM标注)
```

## 数据管线

```
乐谱(.krn/.mxl) → LLM和声分析 → chord_progressions → 训练 ChordGPTv4
                 → LLM旋律分析 → melody_notes       → 训练 MelodyGPT
```

- **和声标注**：DeepSeek V4 Flash 从结构化乐谱 JSON 一步产出完整和声分析（461首，31651和弦）
- **旋律标注**：同样使用DeepSeek V4 Flash

## 模型演进

| 版本 | 架构 | 参数 | 准确率 | 关键创新 |
|------|------|------|--------|---------|
| V1 | 2层 LSTM | 233K | 50.1% | 功能级+和弦级分层 |
| V2 | 6层 GPT | 6.4M | 49.0% | 长序列生成 |
| V3 | 6层 RichGPT | 6.4M | 64.0% | 8维富 token |
| V4 | 12层 GPT | 25.2M | 84.1% | 结构感知 |
| V5 | 16层 d=768 | 152M | 93.5% | 6维和声表示 |

## 核心创新

1. **结构化功能表示**：将传统和声分析中的语义变量（功能、时值、拍位、转位、终止式）显式编码为 6 维生成 token
2. **和声优先生成**：先生成严格古典功能和声，再生成旋律（harmony-driven melody generation）
3. **理论引导解码**：乐理评分函数（scorer）在采样过程中引导模型，而非事后修正
4. **完整符号化管线**：composition → arrangement → performance

## 已知限制

- 旋律模型存在自回归退化（和弦音间振荡），已通过 scorer 引导 + 反振荡拦截缓解，但是旋律的生成效果依旧很不好
- Cadence 头训练失效，改用 D/PD→T 规则分句
- 仅支持 C 大调

## License

MIT

---

# 📈 迭代进展（2026-09-11 接手后）

> 本节在原版 README 之后按**迭代顺序**追加，记录一步步的过程。
> 细节见：
> [02-新工作进展与结果.md](02-新工作进展与结果.md)（技术日志，按轮次）
> · [03-全项目技术梳理.md](03-全项目技术梳理.md)（结构化总览与全部数字）
> · [04-工作历程报告.md](04-工作历程报告.md)（工作历程复盘）

## 迭代 1：先建评估体系（否则不知道改动方向）

原项目只有准确率。先建立两条证据链：

- **乐理合格度**（`src/metrics/theory_metrics.py`，MusicAIR 口径）：
  key confidence（Krumhansl-Schmuckler）、melodic smoothness（级进率等）、
  rhythm matching（强拍对齐）；
- **防模仿证据**（`src/metrics/anti_imitation.py`，MusicLDM + DRMW 口径）：
  SIMAA@90/95、17 维特征马氏距离 + Welch t 检验。

一跑基线，旧 MelodyGPT 的问题立刻量化：马氏距离 **8.69**（逼近 DRMW
"≥10=随机"判定线）、和弦音吻合率仅 0.414。

## 迭代 2：旋律模型非自回归化（v1 → v4）

自回归的两个顽疾（和弦音间振荡、Cadence 头失效）是机制问题，改用
**GETMusic 式 D3PM 离散扩散**（`src/model/melody_diffusion.py`）：

- [MASK] 吸收态 + condition flags；双向 Transformer + 双头；
- 理论 scorer 解码侧引导（引导而非事后修补）；
- 训练 100 epoch 仅 7 分钟（6.2M，本地 4060）。

同条件对比全面超过自回归基线（级进率 0.54→0.77、和弦音吻合 0.41→0.82、
振荡降到 1/3）。期间发现 **14M 大模型恢复率更高但会背谱**（SIMAA@90=0.15）
——防模仿指标第一次用于模型选型，最终定 6.2M。

## 迭代 3：修复训练/推理一致性（v5）

接入完整管线后，输出振荡率 46%（评估集同一模型仅 2%）——查出根因：
**语料和声逐音变化（98%），管线却按和弦块重复 4 次喂条件**。
原版 MelodyGPT 有一模一样的写法，很可能是其振荡顽疾的深层原因。
修复后半小节条件块 + 类型重映射 + 同音衰减：振荡 46%→23%，
级进率 0.51→0.61。

## 迭代 4：全局结构（v6）与乐句（v7）

- **v6**：位置编码外推失效（训练只见过 16 音的位置嵌入，生成 192 音时
  后面是随机嵌入）→ 换 **RoPE 相对位置** + 结构流（全曲位置/距尾距离），
  级进率 0.590→0.753；
- **v7**：用本地 Ollama（`think: false` 是关键坑）标注 579 首乐句边界
  → 乐句流。模型**自发学会句末拉长**（4.35 vs 句中 3.85，人类语料同模式
  4.13/3.74），结尾命中主音 38%→88%。

## 迭代 5：句子级数据与模型（v8）

诊断出根本问题：**旧语料是 23.5 音的单乐句片段，模型从未见过完整句子**。
换用 music21 离线自带巴赫众赞歌，构建 **368 首句子级数据集**
（`build_chorale_dataset.py`，终止式理论标注）+ v8 模型
（`train_v8_chorales.py`：终止式流 + 边界预测头，结构流防泄漏）：

- 32 音窗口（≈2-3 个乐句），恢复准确率 **0.857**；
- **边界 F1 0.08-0.35**（高精确/低召回）——"模型能否自己划句"的诚实起点；
- 本地 Ollama 审计发现两套标注是**层级差异**（终止式=乐句层 3.4 句/首，
  LLM=动机层 6.2 句/首），这是下一步层次化标注的依据。

两级生成（`gen_sentence.py`）：规划器从和声推导终止式计划 → 扩散模型
实现 → 三轨 MIDI。乐句长度 11.5 vs 语料 11.4、前后句关系 0.51 vs 0.57。

## 迭代 6：听感工程（反馈驱动的六轮修复）

| 反馈 | 根因 | 修复 |
|------|------|------|
| "诡异" | 区段式八度拱形（37% 整八度大跳） | 跳进最小化八度分配 |
| "拍号对不上" | 和弦时值非节拍化 → 三轨网格错位 | 4/4 小节吸附 + 八分网格量化 |
| "不会开始结束" | 位置外推 + 无全局结构 | RoPE + 结构流 + 终止式公式 |
| "句子要喘气" | 无乐句级结构 | 乐句流 + 句末气口 |
| "两个都是小调" | 数据大小调近半半 + 调式倾向弱 | `--mode` 过滤 + 调式一致性评分 |
| "织体没了/高音区" | 重构丢轨 + 八度基准高一个八度 | 补回三轨 + 基准 72→64 |

## 当前版本速览

### 怎么跑（最新）

```bash
pip install -r requirements.txt

# 模型权重 (Releases, 不随仓库分发): 下载后放到项目根目录
curl -L -o melody_diffusion_v8_chorales.pt   https://github.com/MoveBricksWorker/classical-music-ai-experiments/releases/download/model-v8/melody_diffusion_v8_chorales.pt

# 句子级生成 (主入口): 众赞歌和声框架 + 终止式计划 + v8 扩散模型 → 三轨 MIDI
python gen_sentence.py --mode major --seed 5

# 训练句子级模型 (本地 8GB GPU, ~15 分钟)
python train_v8_chorales.py --epochs 120 --batch 64 --d 320 --layers 6 --struct-drop 0.5

# 评估
python analyze_syntax.py       # 句法: 边界F1/终止式实现率/乐句长度/前后句关系
python evaluate_report.py      # 乐理 + 防模仿证据链
```

### 模型演进（全部本地 RTX 4060 8GB 训练）

| 版本 | 关键改进 | 恢复准确率 | SIMAA@90(背谱检测) |
|------|----------|-----------|--------------------|
| v1 | 非自回归扩散基础版 | 0.736 | 0.000 |
| v2 | 14M 容量实验 | 0.902 | 0.150 ✗背谱 |
| v3 | +拍位条件 (14M) | 0.924 | 0.217 ✗ |
| v4 | +拍位条件 (6.2M) | 0.746 | 0.017 |
| v5 | +块恒定条件 (修训练/推理不一致) | — | 0.000 |
| v6 | +结构流+RoPE | 0.625* | 0.000 |
| v7 | +乐句流 | 0.822 | 0.000 |
| **v8** | **+终止式流+边界预测头 (句子级)** | **0.857** | — |

*恢复准确率随条件欠定程度变化，非直接可比。

> **模型权重不放在仓库里**，从 Releases 下载：
> [`melody_diffusion_v8_chorales.pt`](https://github.com/MoveBricksWorker/classical-music-ai-experiments/releases/download/model-v8/melody_diffusion_v8_chorales.pt)
> （model-v8 release，28MB）；消融系列（v1-v7）与全部中间产物见本地备份，
> 各版本结论见 [03-全项目技术梳理.md](03-全项目技术梳理.md)。

---

# 🔧 迭代 7：评估修订（2026-09-11 第二轮）

> 外部评审指出三个问题，本轮全部修复。完整报告见
> [05-评估修订报告.md](05-评估修订报告.md)。

## 7.1 验证集泄漏 —— "恢复准确率 0.857"是虚高

滑窗 `win=32/stride=6` 使同曲相邻窗口重叠 81%，而旧划分是样本级随机 90/10
**没有按曲分组**：135 个验证窗口里 **129 个（96%）与训练窗口同曲且时间重叠**。

同一份代码、同一套超参，**只改划分**的对照实验：

| 配置 | 恢复 t50 | 边界 F1±1 | 全掩码生成 |
|------|----------|-----------|-----------|
| 旧口径（样本级随机，泄漏） | **0.809** | 0.604 | **0.727** |
| **按曲分组（诚实）** | **0.412** | 0.607 | 0.320 |

更干净的**泄漏分解实验**（`analyze_leakage.py`：同一个模型、同一份训练集，
只换验证窗口）给出泄漏的纯膨胀量：

| 验证窗口 | 恢复 t50 | 全掩码生成 | 边界 F1@.5 |
|----------|----------|-----------|-----------|
| 同曲未训窗口（旧口径） | **0.803** | **0.702** | 0.733 |
| 未见过的曲（诚实口径） | **0.412** | **0.326** | 0.487 |

即：**旧口径的恢复率虚高约 +0.39、生成虚高约 +0.38、边界 F1 虚高约 +0.25**。

v8 权重按原口径复核为 0.738（文档 0.857，不可复现）；它在我这边"干净 val"
上的 0.897 其实是**训练窗口自评**（v8 训练覆盖 363/368 首的全部窗口）。

**5 折按曲交叉验证（新主模型 v9）**：

| 指标 | 均值 ± 标准差 |
|------|---------------|
| 恢复 t=0.5 | **0.407 ± 0.021** |
| 边界 F1（±1 容差） | **0.532 ± 0.030** |
| 边界 F1（严格 @0.5） | 0.491 ± 0.032 |
| 全掩码生成 | 0.264 ± 0.050 |
| **自洽边界 F1**（模型给自己写的旋律划句） | **0.095 ± 0.034** |
| 终止式实现率 | 0.344 ± 0.079（人类 0.53） |
| 防模仿 SIMAA@90 | 0.000（人类 0.000） |
| 马氏距离 | 4.0–5.1（人类 3.98，随机线 10） |

## 7.2 渲染层曾把模型学到的节奏全部压掉

旧 `write_midi` 把每个旋律音截断到 ≤1 拍、句末再 ×0.65 —— 实测旧成品
`major_1/2.mid` **末音长音 0%**，即"句末拉长"从未进入 MIDI。
新渲染在小节内按节奏 bin 比例分配、吸附十六分网格、时值取相邻起点差
（不重叠）、句末留气口：

| | 旧渲染 | 新渲染 | 人类语料 |
|---|---|---|---|
| 旋律时值 | 全部 ≤1.00 拍 | 0.50–3.75 拍 | — |
| 末音长音 | **0%** | **100%** | 64% |

旧口径保留为 `--render slot` 以便复现历史成品。

## 7.3 组成来源（哪些是生成 / 检索 / 规则）

| 组件 | 来源 |
|------|------|
| 和声框架（func/type/root） | **语料检索**（采一首众赞歌的逐音和声，4 音块恒定） |
| 乐句/终止式计划 | **语料检索**（该曲真实乐句边界与终止式类型） |
| 旋律音高与节奏 | **模型生成**（v9 非自回归扩散） |
| 终止音修正 / 八度分配 / 三轨织体 / 气口与网格 | **规则** |

因此"乐句长度 11.5 vs 语料 11.4 ✓"是**计划本身**的属性，不是模型学出来的
（本轮用同一批窗口配对实测：生成 8.969±0.253 vs 真值 8.969±0.253，**完全相同**）；
模型侧真正的句法指标是**自洽边界 F1 = 0.095**（当前最大短板）。

## 7.4 新增文件与用法

```bash
# 主模型 v9 (按曲分组 90/10 训练) → 三轨 MIDI
python gen_sentence.py --mode major --seed 5            # 默认已优先使用 v9
python gen_sentence.py --pieces 3 --render rhythm        # 生成 3 首, 节奏渲染

# 训练 (按曲分组; --folds 5 交叉验证)
python train_v9_chorales.py --folds 5 --epochs 120 --bw 0.5 --bw-pos-weight 5

# 完整证据链 (恢复/边界/生成/自洽/终止式/防模仿, 含人类基线)
python evaluate_chorale_model.py --model melody_diffusion_v9_chorales.pt

# 泄漏分解 (同曲未训窗口 vs 未见曲)
python analyze_leakage.py

# 渲染契约回归测试
python tests/test_rendering.py
```

- `src/chorale_conditions.py`：条件流编码 + 按曲分组划分的**唯一实现**
  （此前训练/生成/评估三处手抄，迁移已验证与原实现逐窗口一致）
- `train_v9_chorales.py` / `evaluate_chorale_model.py` / `analyze_leakage.py`
- v9 权重走 Releases：[**`model-v9`**](https://github.com/MoveBricksWorker/classical-music-ai-experiments/releases/tag/model-v9)
  —— 主模型 [`melody_diffusion_v9_chorales.pt`](https://github.com/MoveBricksWorker/classical-music-ai-experiments/releases/download/model-v9/melody_diffusion_v9_chorales.pt)
  （28MB）+ 5 折权重 `melody_diffusion_v9_fold{0..4}.pt` + 两个对照权重
  （`v9_leaked.pt` 复现旧口径 0.809、`v9_grouped_v8hp.pt` 复现诚实口径 0.434）

## 7.5 一句话

**尺子比读数重要**：0.407 比 0.857 更有用。语法（和声）仍是检索来的，
句子（乐句）现在有了可稳定测量的边界指标（0.53±0.03），
但模型对自己写的句子还没有句法感（自洽 0.095）——这是下一阶段的主攻方向。

---

# 📚 迭代 8：四声部数据与织体标注（2026-09-12）

> 目标：解决 v9 的数据瓶颈（**18.5k 个单声部音符**——只用了众赞歌 4 个声部里的 1 个）。
> 完整说明见 [06-数据集与织体标注.md](06-数据集与织体标注.md)。

## 8.1 数据规模：18.5k → 66.9 万音符

| 数据集 | 来源 | 曲目 | 纵向切片 | 音符 | 织体分布 |
|--------|------|------|----------|------|----------|
| `chorales_satb_v2.json` | music21 巴赫众赞歌（公开领域） | 368 | 30,095 | **85,699** | 64% 柱式 |
| `palestrina_satb_v1.json.gz` | music21 Palestrina 全集（1318 个 .krn） | 1,236 | 270,506 | **583,707** | 77% 复调/模仿 |

- 众赞歌改用**全部四个声部**（此前只用 Soprano）；Soprano 音符数 18,547 与 v1
  **完全一致**——可比性校验通过，v9 的乐句/终止式标注可无缝对照；
- Palestrina 是免费、离线、无版权风险的**对位语料**，与 SATB 众赞歌**同 schema**
  （只多 `style` 字段），可"先复调预训练 → 再众赞歌微调"；
- 归档仅 3.1 MB（gzip），可离线完整复现。

## 8.2 织体标注（本次新增）

三级标注：**切片级**（发声音数/起音/同音级/八度加倍）·
**小节级伴奏音型**（柱式/长音/阿尔贝蒂/破碎/琶音——后三者是为下一步钢琴语料预留的检测器）·
**乐句级织体类型**（柱式/模仿/复调/持续音/齐奏/混合），
外加 5 个客观特征（同起音率、节奏独立性、模仿串长、持续音占比、齐奏率）。

## 8.3 审计发现：LLM 打标的真实能力边界

用本地 Ollama（`qwen3.5:9b-q4_K_M`）复核 45 条分层抽样：**一致率仅 29%**——
它从未输出 `imitative`（动机级分析超出能力），把 64% 柱式的众赞歌判成 45% polyphonic，
却全程给出 0.80–0.95 的置信度。

**结论**：不要用 LLM 做大规模织体打标；特征值作为主标注（客观可复现），
硬标签待人工金标准（100–200 句、双人标注）校准。

**但审计流程本身有效**：首轮 0% 一致率追查下去，抓出了管线里的真 bug
——乐句记录把"句末切片"存成了 `start_slice`，导致喂给模型的表只有一行。
（与项目既有经验一致：**管线本身要先验证**，否则会误读成"LLM 不懂音乐"。）

## 8.4 下一步

1. **v10 四声部模型**：先做最易见效的设定——"给定三个声部、用扩散模型补第四个"
   （masked infilling，与现有 D3PM 框架兼容）；
2. **对位预训练**：Palestrina 预训练 → 众赞歌微调（加 `style` 条件流）；
3. **织体条件化生成**：把 `texture` 作为条件流接进生成器，替换固定的三条织体规则；
4. **扩钢琴语料**：music21 自带的莫扎特奏鸣曲 / 海顿 / 舒伯特 / 舒曼
   （约 60 首，可激活 `alberti`/`broken`/`arpeggio` 检测器）。
