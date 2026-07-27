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

### 安装

```bash
pip install -r requirements.txt
```

### 下载模型权重

从 [Releases]下载以下文件放到项目根目录：

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
- **旋律标注**：阿里云百炼免费 API（Qwen 模型轮换），28912 旋律音，8 种角色分类

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

