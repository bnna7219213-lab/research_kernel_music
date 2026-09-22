# 阶段 A 执行手册（K8s + 8×A100-80GB 版）

> 版本：v0.1（2026-09-22）
> 关联：roadmap_v0.1.md → 阶段 A（第 1–4 周）
> 前提：开发机为笔记本（RTX 4050），训练机为 K8s 8×A100-80GB，任务通过 kubectl 提交。

---

## 1. 总览

```
[当前笔记本 / dev box]                [K8s 8×A100 集群]
   ┌─────────────────┐   kubectl apply  ┌──────────────────┐
   │ 代码开发 & 评审  │ ───────────────▶ │ 训练 / 推理任务  │
   │ 数据预处理脚本  │   rsync / PV     │ 挂载持久卷(PV)    │
   │ 小批量 debug    │                  │ 产出 checkpoint   │
   └─────────────────┘                  └──────────────────┘
```

**核心策略**：所有需多卡的任务提交至 K8s；本机只跑数据预处理小任务和代码评审。

---

## 2. 分周任务

### 第 1 周：环境镜像 + 数据集 v0

| 编号 | 任务 | 交付物 | 位置 | 负责 |
|---|---|---|---|---|
| W1.1 | 构建 Docker 镜像（PyTorch 2.x + cu12 + 依赖） | `docker/Dockerfile` 通过 CI 构建 | `docker/` | 工程 |
| W1.2 | 编写数据集下载脚本（MusicCaps + FMA 子集） | `scripts/download_datasets.sh` 可执行 | `scripts/` | 数据 |
| W1.3 | 编写数据指纹工具 | `scripts/compute_data_fingerprint.py` 可用 | `scripts/` | 工程 |
| W1.4 | 核验 K8s 集群 PV 挂载点与配额 | 文档化路径、容量 | `k8s/storage_pvc.yaml` | 工程/运维 |

**验收**：镜像在 K8s 可拉取；MusicCaps 成功下载 ≥100 条样例；数据指纹可复现同一 MD5。

### 第 2 周：推理复现 (YuE2-3B + ACE-Step v1)

| 编号 | 任务 | 交付物 | 基线 |
|---|---|---|---|
| W2.1 | 拉取 YuE2-3B 权重与推理代码，硬编码 8 卡 | 推理脚本可跑，出 ≥20 条 demo 音频 | YuE2-3B（AR+NAR+FM） |
| W2.2 | 拉取 ACE-Step v1 权重与推理代码 | 推理脚本可跑，出 ≥20 条 demo 音频 | ACE-Step v1 (DiT+DCAE) |
| W2.3 | 编写盲听评分表模板 | `src/eval/blind_listening/` 中生成对比播放列表及评分表 | 评估 |

**验收**：两个基线推理在 8 卡 A100 上跑通；评分表 ≥10 人 × 30 prompts 完成。

### 第 3 周：训练 pipeline 跑通 (MusicCaps 子集)

| 编号 | 任务 | 说明 |
|---|---|---|
| W3.1 | 跑通 DCAE tokenizer 训练 | 8 卡，5k 条 10s 音频，目标重建 FAD ≤ 开源 |
| W3.2 | 跑通 DiT/FM 渲染器预训练 | 8 卡，30k 条 10s，REPA 语义对齐 |
| W3.3 | 输出 W&B 实验记录 | 训练曲线、显存、吞吐量、checkpoint |

**验收**：训练能稳定跑到 10k 步不 OOM；checkpoint 可恢复；W&B 对比基线清晰。

### 第 4 周：评估体系 v1 + 基线对齐

| 编号 | 任务 | 交付物 |
|---|---|---|
| W4.1 | 自动化 FAD / CLAP 评估脚本 | `src/eval/compute_fad.py`, `compute_clap.py` |
| W4.2 | 基线锚点文档终版 | `docs/baseline_anchor.md`（含内部 MOS） |
| W4.3 | 阶段 A 评审 & Go/No-Go 决策 | 评审纪要 |

**评估标准**：
- 重建 FAD ≤ 开源 EnCodec/DCAE 同条件的 1.1×
- 生成音频 CLAP score 与 MusicGen 对齐（±0.02）
- 内部盲听 MOS ≥ 3.5/5

---

## 3. 数据集选择（首版固定：MusicCaps + FMA）

| 数据集 | 用途 | 规模 | 许可 | 下载方式 |
|---|---|---|---|---|
| **MusicCaps** | 训练 / 评估对齐 | 5,500 条 × 10s + 人工 caption | CC BY-SA 4.0 | YouTube 自取（脚本提供） |
| **FMA medium** | 数据扩充 / 弱标注 | 25,000 首全曲, 16 类曲风 | CC 系逐曲 | 官方 archive.org（离线） |
| **MagnaTagATune** (可选) | 多标签辅助 | 25,000 × 29s + 多标签 | CC | GitHub 镜像 |

> **为什么选这两个**：
> - MusicCaps 是音乐 caption 对齐的 benchmark，有文本监督信号，适合首版快速对齐 text → music
> - FMA medium 规模适中、许可明确，可验证 codec 与渲染器在弱标注数据上的泛化

> **⚠️ 不在范围**：MTG-Jamendo（非商用，不进入训练集）；商用数据后续另立采购清单。

---

## 4. 验收清单（Go/No-Go 评审用）

```
[ ] W1: Docker 镜像在 K8s 可调度
[ ] W1: MusicCaps 至少 100 条下载并生成数据指纹
[ ] W2: YuE2-3B 推理产出 ≥20 条 demo
[ ] W2: ACE-Step v1 推理产出 ≥20 条 demo
[ ] W2: 盲听评分 ≥10 人 × 30 prompts 入库
[ ] W3: DCAE 训练到 10k 步不 OOM，checkpoint 可恢复
[ ] W3: 渲染器训练到 10k 步，W&B 记录完整
[ ] W4: FAD/CLAP 评估自动化脚本可用
[ ] W4: 基线锚点文档完成（YuE2-3B / ACE-Step v1 / MusicGen / Suno 对比）
[ ] W4: 评审纪要：M-B1 Go / No-Go 决策
```

---

## 5. 关键依赖与风险

| 依赖 | 状态 | 兜底 |
|---|---|---|
| K8s 集群可调度 8×A100 | △ 待运维确认 | 如 8 卡同时难排，先用 4 卡做 W1–W3 |
| MusicCaps YouTube 可访问 | △ 部分区域受限 | 用 yt-dlp 多镜像；准备国内中转或已下载源 |
| YuE2-3B 权重 & 代码可用 | 需法务审查 NC 许可 | 若不可用，改用 ACE-Step v1.5 做旋律生成，用 YuE 开源代码做参考 |
| W&B / MLflow 接入 | 需企业 license | 先用本地 MLflow 离线追踪，不依赖外网 |

---

## 参考

- roadmap_v0.1.md → 阶段 A 总体说明
- plan.md §2 技术格局 & §6 算力
- ACE-Step v1 仓库（Apache 2.0）：github.com/ACE-Step/ACE-Step
- YuE2-3B 模型卡（m-a-p）
- MusicCaps：github.com/nils-wagner/MusicCaps
- FMA：github.com/mdeff/fma
