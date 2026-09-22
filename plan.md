# 自研 T2Music（Text-to-Music）模型技术规划

> 版本：v0.1（2026-09-22）
> 定位：探讨从零自研 T2Music 模型的技术路线，给出选型结论、数据/算力方案与分阶段里程碑，供立项评审使用。

---

## 1. 目标与范围

**目标**：构建一个可从文本描述（风格/情绪/场景标签，可选歌词）生成长达数分钟、具有商业可用音质的音乐/歌曲生成模型。

**范围内**：
- 纯器乐生成（text-to-music）与歌曲生成（lyrics-to-song）两条子任务线
- 自定义音频 tokenizer / 潜空间编码器、生成主干、文本条件注入
- 数据管线、训练框架、评估体系

**范围外（后续单独立项）**：
- TTS / 语音合成、纯符号域（MIDI/ABC）作曲
- 具体产品化（App/SDK、流媒体分发）与商业化谈判

**关键假设**：
- 团队具备深度学习训练工程能力（多卡分布式、数据清洗、推理优化）
- 优先开源可复现路线；商用合规作为硬约束纳入数据选型
- 初期算力规模为"学术/中型团队级"（数十至数百张 A100/H100 级别或等效云租用），而非大厂万卡级

---

## 2. 技术格局（2025–2026 现状速览）

### 2.1 代表性系统

| 系统 | 团队 | 架构范式 | 时长/能力 | 许可 | 备注 |
|---|---|---|---|---|---|
| MusicGen | Meta (2023) | AR LM + EnCodec 多码本 | ~30s 器乐 | 非 商用权重 | 奠基性工作，RAFT 延迟码本并行 |
| Stable Audio Open | Stability AI (2025) | 潜空间扩散（DiT） | 短片段/音效 | 开放权重 | CC 数据训练 |
| AudioLDM 2 | 开源学术 | 潜空间扩散 | 通用音频 | 开源 | 统一语音/音乐/音效框架 |
| YuE | HKUST / M-A-P (2025) | LLaMA2 AR，双轨（人声/伴奏）解耦下一 token 预测 | 5 分钟整曲+歌词对齐 | Apache 2.0 | 首个开源全曲歌词到歌曲基础模型，万亿 token 训练 |
| YuE2 | M-A-P (2026-09) | AR–NAR Mixture-of-Transformers + Flow Matching + 符号乐谱规划（ABC） | 全曲 + 可编辑乐谱 | 权重 CC BY-NC 4.0 | WildSongBench 上与 Suno v5/v6 同档；3B 权重 24GB 单卡可推理 |
| ACE-Step v1 | ACE Studio + StepFun (2025) | Flow Matching + 深度压缩自编码器（DCAE）+ 线性 Transformer（3.5B） | 4 分钟 / A100 约 20s | Apache 2.0 | 速度比 AR 基线快 15×，REPA 语义对齐加速收敛 |
| ACE-Step v1.5 | 同上 (2026) | 混合 LM 规划器（Qwen3-0.6B）+ DiT 渲染器，FSQ 5Hz 离散码 + DMD2 蒸馏 4–8 步 | 10 分钟 / A100 <2s | MIT | LM 生成歌曲蓝图（YAML），内在 RL 对齐，50+ 语言 |
| DiffRhythm | 西北工大 (2025) | 潜空间扩散全曲生成 | 全曲 4m45s / 10s 生成 | 开源 | 证明纯扩散路线可做全曲歌曲 |
| LeVo / LeVo 2 | 腾讯 (2025/2026) | LM（LeLM）+ Music Codec，分层表征 + 渐进后训练（DPO 多偏好对齐） | 全曲歌曲 | — | 开源榜单前列，接近商用系统 |
| SongBloom | (2025) | 交错式 AR 草图 + 扩散精修 | 全曲 | 开源 | AR 与扩散混合的代表 |
| HeartMuLa | (2026) | HeartCLAP + HeartTranscriptor + HeartCodec（12.5Hz 低帧率）+ LLM 生成器 | 6 分钟，3B/7B | 开源 | **首次证明学术级数据+算力可复现 Suno 级系统** |
| Suno / Udio / Mureka | 商业闭源 | 未公开 | 全曲 | 闭源 | 当前商用标杆（Suno v5/v6） |

（来源：arXiv 2503.08638 YuE、arXiv 2506.00045 ACE-Step、arXiv 2601.10547 HeartMuLa、YuE2 模型卡、各项目 GitHub）

### 2.2 三大架构范式对比

**A. 自回归离散 token（AR-LM over codec）**
- 思路：音频 → 离散 codec token（EnCodec / 自研语义+声学双码本）→ Transformer 下一 token 预测
- 优点：歌词对齐强、天然支持长上下文与 in-context 学习（风格迁移/翻唱）、可借用 LLM 生态（LoRA、RLHF/DPO）
- 缺点：推理慢（逐 token）、误差累积导致长曲结构漂移；多码本延迟并行策略复杂
- 代表：MusicGen、YuE、LeVo、HeartMuLa

**B. 潜空间扩散 / Flow Matching（连续潜变量）**
- 思路：音频 → 压缩潜空间（DCAE/VAE）→ DiT 用 flow matching 一次性并行去噪
- 优点：推理极快（蒸馏后 4–8 步）、音质上限高、训练收敛快（可挂 REPA 语义对齐）
- 缺点：长程结构一致性弱、歌词对齐需要额外机制、编辑/可控性需专门设计
- 代表：Stable Audio Open、ACE-Step、DiffRhythm

**C. 混合式：符号/语义规划 + 声学渲染（当前主流收敛方向）**
- 思路：小 LM 做高层"作曲规划"（结构、和弦、旋律、蓝图），生成模型做高保真渲染
- 优点：兼顾结构连贯性与音质，规划结果可编辑（乐谱/蓝图），可解释性强
- 缺点：两阶段训练与对齐复杂度高
- 代表：ACE-Step 1.5（LM 蓝图 + DiT）、YuE2（ABC 乐谱 CoT + flow matching）、SongBloom（AR 草图 + 扩散精修）

**结论：业界已明显收敛到范式 C**——"可编辑的结构规划 + 快速的高保真渲染"，同时 codec 朝低帧率（5–12.5Hz）演化以降低序列长度。

---

## 3. 自研技术路线选型

### 3.1 推荐主线：混合式（C 路线）

**总体架构**：

```
用户输入（风格描述 / 歌词 / 参考音频）
        │
        ▼
[阶段1] 规划器 Planner（1B–4B LM，可基于开源 LM 微调）
   · 结构规划：intro/verse/chorus/bridge 段落、BPM、调性
   · 旋律/和弦符号规划（ABC 或 YAML 蓝图），可编辑、可回传修改
        │
        ▼
[阶段2] 渲染器 Renderer（DiT + Flow Matching，1–4B）
   · 输入：蓝图 + 风格/歌词 embedding + 段落级时间轴
   · 输出：压缩潜空间（自研 DCAE/VAE，低帧率）
        │
        ▼
[阶段3] 解码器 Decoder（DCAE decoder + vocoder）
   · 潜空间 → mel → 48kHz 立体声波形
```

**关键设计决策**：

1. **音频表征**：自研低帧率 codec（目标 6–12.5Hz、48kHz 立体声）。
   - 参考方案 A（离散）：语义编码器（Whisper/WavLM/MuEncoder 级特征）+ 量化（FSQ/RVQ）+ flow-matching 解码，参考 HeartCodec、MuCodec（0.35kbps@25Hz 先例）
   - 参考方案 B（连续）：深度压缩自编码器（时间压缩 8×，如 ACE-Step 的 DCAE f8c8 + ADaMS-HiFiGAN vocoder），渲染侧用 flow matching
   - 建议先做**连续潜空间 + Flow Matching**（收敛快、工程量小），离散 codec 作为第二阶段替代/融合项

2. **语义加速收敛**：训练时用 MERT/mHuBERT 做 REPA 表征对齐（ACE-Step 验证可大幅加速收敛）。

3. **歌词对齐**：歌词经 Conformer/轻量编码器注入，配合注意力对齐奖励（ACE-Step 1.5 的 AAS 内在奖励可参考）；后期用 DPO 做偏好对齐。

4. **规划器不必从零训**：基于开源 LM（Qwen3-0.6B~4B 级）SFT 到"歌曲蓝图生成"任务，成本可控；其符号输出（ABC/结构标签）天然支持用户编辑与 Agent 改写闭环（YuE2 已验证该产品形态）。

### 3.2 备选路线与何时切换

| 情形 | 切换建议 |
|---|---|
| 歌词-人声对齐始终不达标 | 转双轨解耦 AR（YuE 式 track-decoupled next-token prediction）或 SongBloom 式 AR+扩散混合 |
| 算力严重不足（<16 卡） | 先做纯器乐 + 短片段（30–60s）的潜扩散模型验证核心链路，再扩展 |
| 目标偏向可控编辑产品 | 优先加大规划器权重，蓝图为产品中心（ACE-Step 1.5 / YuE2 形态） |

### 3.3 与开源基线的关系

- **不强求从零复现全部组件**：首阶段可直接复用 ACE-Step v1（Apache 2.0）的 DCAE/vocoder 与代码框架做自研改造，加速验证；商业目标版本再替换为完全自研权重
- 对标基线固定为：YuE2（开源榜第一）、HeartMuLa-3B、LeVo 2，评估指标用 WildSongBench/SongBench 协议

---

## 4. 数据方案

### 4.1 公开数据集盘点

| 数据集 | 规模 | 标注 | 许可 | 用途 |
|---|---|---|---|---|
| MusicCaps | 5.5k 条 10s 片段 | 音乐家撰写 caption | CC BY-SA 4.0（音频需自取自 YouTube） | 文本条件对齐、评估 |
| MTG-Jamendo | 5.5 万+ 全曲，195 标签 | genre/instrument/mood 标签 | **非商用研究专用**（CC BY-NC-SA，另有 Jamendo 商用授权需付费） | 标签对齐预训练；商用需购授权（Jamendo v. NVIDIA 诉讼 2026-06 佐证其执行意愿） |
| FMA | 10 万+ 曲目（large ~10.6 万） | 曲风标签 | CC 系（逐曲不同） | 大规模无标注/弱标注预训练 |
| Jamendo / Free Music Archive / 公共领域曲目 | 百万级 | 弱标注 | 逐曲 CC | 扩规模主力 |
| SongEval / MusicEval / TTM-Bench | 评估集 | 人工评分 | 开放 | 评估 |

### 4.2 自动标注管线（核心投入点）

1. **音频清洗**：响度归一（loudness match）、去人声/伴奏分离（Demucs 级模型）、纯器乐/含人声分流、掐头去尾、去重复采样
2. **元数据重建**：
   - 歌词：歌词识别模型（可微调 HeartTranscriptor 类方案或 ASR+对齐）
   - 风格标签：用 CLAP/MTT（MusicTaggingTransformer）自动打 genre/instrument/mood 标签
   - 描述性 caption：用音频理解 LLM（Qwen-Audio、MuMu-LLaMA 类）批量生成 caption，再用 CLAP 分数过滤低置信样本
   - 结构标注：段落切分（intro/verse/chorus）用结构检测模型 + 统计平滑
3. **蓝图/乐谱监督**：对含人声曲目做旋律转录（SheetSage2 类）与和弦估计，生成（歌词, ABC 乐谱, 音频）三元组，用于规划器 SFT
4. **质量分层**：L1 高质量（人工/权威标注）→ L2 自动标注高置信 → L3 弱标注海量数据，训练时按层配比采样

### 4.3 规模目标

- PoC 阶段：10 万–50 万条 30–60s 切片（约 2k–10k 小时）
- 中规模：50 万–200 万小时级累计音频（参考 HeartMuLa"学术级数据"量级；YuE 为万亿 token ≈ 数十万小时）
- 合规底线：只收录许可明确的数据（自有授权、CC 允许、免版税/公共领域、MIDI 合成数据）；MTG-Jamendo 仅限内部研究验证，不得进入商用训练集

---

## 5. 训练管线

### 5.1 阶段拆分（对齐推荐架构）

1. **P0 Tokenizer 训练**
   - DCAE/vocoder：重建损失（mel L1 + 对抗 + 特征匹配），目标 PESQ/重建 FAD 达到 ACE-Step 同级
   - 数据：任意许可干净音频 1k–10k 小时
2. **P1 渲染器预训练**（主力算力消耗）
   - Flow matching 目标（CFG/蒸馏后处理），条件 = 文本 tag embedding（UMT5/T5 级）+ 歌词 embedding + REPA 语义对齐损失
   - 32k token 有效上下文起步，逐步扩到全曲
3. **P2 规划器 SFT**
   - 基于 0.6B–4B 开源 LM，训练 (caption/歌词 → 蓝图/ABC) 映射；数据来自 §4.2-3 的转录产物 + 商业蓝图合成
4. **P3 联调与对齐**
   - 规划器 + 渲染器端到端联训（LoRA 起步）；歌词对齐用内在奖励（注意力对齐分数）+ DPO 偏好对齐
5. **P4 蒸馏与推理优化**
   - DMD2/一致性蒸馏将 50 步降到 4–8 步；CUDA Graph、FlashAttention、量化推理，目标消费级显卡可运行

### 5.2 工程栈

- 训练：PyTorch + FSDP/DeepSpeed ZeRO-2/3，bf16，FlashAttention-2/3
- 数据：WebDataset/流式读取，预处理离线化（tokenize 与特征一次算好）
- 评估内环：FAD（Frechet Audio Distance）、CLAP score、歌词 PER（词错误率）、SongBench 客观分 + 每里程碑固定盲听（MOS 10–30 人）
- 版本管理：数据/权重/配置全部带指纹，实验用 W&B/MLflow 追踪

---

## 6. 算力估算（量级参考）

| 阶段 | 模型规模 | 数据量 | 估算 GPU 时（A100/H100 级） |
|---|---|---|---|
| P0 Tokenizer | 100–500M | 1k–10k 小时 | 200–1,000 卡时 |
| P1 渲染器预训练 | 1–4B | 2 万–20 万小时 | 5k–10 万卡时 |
| P2 规划器 SFT | 0.6–4B（LoRA/全参） | 10 万–100 万条蓝图 | 500–5,000 卡时 |
| P3 对齐/联调 | 同上 | 精选 | 2k–2 万卡时 |
| P4 蒸馏 | 渲染器 | 中量 | 1k–1 万卡时 |

- **PoC（验证链路）**：8×A100 跑 1–2 个月（1–3 千卡时）
- **对标 HeartMuLa-3B（Suno 级证明的学术规模）**：约数万卡时
- **对标开源第一梯队（YuE2 / LeVo 2）**：估 10 万卡时以上 + 大规模数据工程投入
- 云租用参考价：A100 约 $1.5–2.5/卡时，即 PoC 约 $2k–8k，中规模 $10 万级

> 以上为数量级估计，实际取决于数据质量、潜空间压缩率与蒸馏方案质量；P1 是最大变量。

---

## 7. 里程碑规划

| 阶段 | 时长（估） | 交付物 | 验收标准 |
|---|---|---|---|
| M0 调研与选型冻结 | 2–4 周 | 本 plan.md 终版 + 复现报告（跑通 YuE2/ACE-Step 推理，内部盲听对齐预期） | 团队对三条路线的取舍达成一致；基线推理环境可用 |
| M1 数据管线 v1 | 4–8 周 | 自动标注流水线 + 10 万条清洗对齐样本 | 标签 CLAP 置信分布达标；抽样 100 条人工核验 ≥85% 准确 |
| M2 Tokenizer（P0） | 4–6 周 | 自研 DCAE/vocoder（或选定复用方案） | 重建 FAD ≤ 开源同级；24s 音频重建盲听不可辨差异 |
| M3 短曲 PoC（P1 首版） | 8–12 周 | 30–60s 器乐 text-to-music 模型 | FAD、CLAP score 与 MusicGen/Stable Audio Open 持平或更好；内部 MOS ≥3.5/5 |
| M4 全曲渲染 + 规划器（P1 扩展 + P2） | 3–4 个月 | 3 分钟级全曲生成，蓝图可编辑 | 歌词 PER 达标；结构连贯性盲听通过率 ≥70%；可编辑乐谱回环演示 |
| M5 对齐与蒸馏（P3+P4） | 3–4 个月 | 蒸馏后 4–8 步推理、风格可控、DPO 对齐 | 速度：A100 全曲 <10s；单卡 24GB 可部署；SongBench/WildSongBench 对标 YuE2 ±10% |
| M6 产品化探索 | 持续 | API/编辑界面、Agent 改写闭环 | 灰度用户留存与生成质量反馈 |

**里程碑间设置"go/no-go"评审**：M3 不达标则按 §3.2 切换路线。

---

## 8. 评估体系

1. **客观**：FAD（music 域）、CLAP score、CLaMP3 score、歌词 PER、节拍/调性一致性（counterfactual 指令遵循测试，2026 年学界已指出 CLAP score 与人类偏好相关性不足，需组合多指标）
2. **主观**：固定 prompt 集（192 条 WildSongBench 式）盲听 MOS，音乐性/人声表现/歌词对齐/结构连贯四维
3. **合规**：输出去重（对训练集的逐字复现检测，参考 YuE §11 的 memorization 检测方法）、AI 水印可选

---

## 9. 风险与应对

| 风险 | 等级 | 应对 |
|---|---|---|
| **数据许可法律风险**（Jamendo 诉 NVIDIA 索赔 1780 万欧元，2026-06） | 高 | 严格许可审计；非商用数据隔离；商用训练只用明确授权数据；法务前置介入 |
| 长程结构连贯性不达标 | 中 | 蓝图/段落条件强约束；必要时切换 AR+扩散混合（SongBloom 式） |
| 歌词-人声对齐弱 | 中 | 注意力对齐奖励 + DPO；双轨解耦表征兜底 |
| 算力/预算不足 | 中 | 分层训练（LoRA、渐进扩容）；优先复用 Apache/MIT 许可组件 |
| 领域迭代快（3–6 个月一代） | 中 | 每里程碑重扫开源榜；架构保留可替换 codec/渲染器的接口 |
| 评估与人类偏好脱节 | 低 | 常态化人工盲听，不单依赖 CLAP 系指标 |

---

## 10. 立即行动项（Next 2 Weeks）

1. 复现推理：部署 YuE2-3B 与 ACE-Step v1.5（Apache/MIT 许可），内部盲听建立质量锚点
2. 数据盘点：梳理团队可合法使用的音频资产与许可清单
3. 算力报价：获取 8×A100 与 64×A100 两档云租用报价，锁定 PoC 预算
4. 指定 1 人负责"许可审计 + 数据指纹"管线（法务协同）

---

## 11. 参考来源

- YuE: Scaling Open Foundation Models for Long-Form Music Generation — arXiv:2503.08638
- ACE-Step: A Step Towards Music Generation Foundation Model — arXiv:2506.00045
- HeartMuLa: A Family of Open Sourced Music Foundation Models — arXiv:2601.10547
- YuE2-3B 模型卡（M-A-P，2026-09，WildSongBench 榜单）— aihub.caict.ac.cn / HuggingFace m-a-p/YuE2-3B
- ACE-Step v1.5 模型卡（ModelScope acestep-v15-turbo-shift3）
- MTG-Jamendo Dataset 官方页 — mtg.github.io/mtg-jamendo-dataset
- MusicCaps（Google，CC BY-SA 4.0）、FMA（CC 系）
- Jamendo 诉 NVIDIA 版权诉讼报道 — Music Business Worldwide, 2026-06-23
- TTM-Bench / MusicEval / SongBench 等评估基准论文
