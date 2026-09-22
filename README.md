# T2Music Phase-A — 自研音乐大模型研究内核

> 企业研发 · 8×A100-80GB · K8s · kubectl 调度

## 目录结构

```
.
├── configs/                 # 训练配置 (YAML)
│   ├── yue2_3b.yaml
│   └── acestep_v1.yaml
├── docker/
│   ├── Dockerfile           # PyTorch 2.5 + CUDA 12.4 + eval 栈
│   └── entrypoint.sh        # 统一 CLI：train/infer/eval
├── k8s/
│   └── submit_train.yaml     # 8 卡 K8s Job 模板
├── scripts/
│   ├── download_datasets.sh  # MusicCaps / FMA 下载
│   └── compute_data_fingerprint.py
├── src/
│   ├── train/               # 训练主入口 + 各适配器
│   ├── eval/                # FAD / CLAP / 盲听
│   └── utils/               # distributed / logger / seed
├── tests/
│   └── smoke_test.py        # CPU-only 开发机冒烟
├── phase_a_exec.md          # 阶段 A 执行手册（K8s 适配版）
├── roadmap_v0.1.md          # 总体路线图
└── plan.md                  # 技术规划
```

## 快速开始

### 1. 本机冒烟（开发机，CPU 即可）

```bash
pip install pyyaml numpy
python tests/smoke_test.py
```

### 2. 提交 K8s 训练（8×A100-80GB）

```bash
# 1. 构建镜像
docker build -t your-registry.io/t2music:phase-a -f docker/Dockerfile .
docker push your-registry.io/t2music:phase-a

# 2. 编辑 k8s/submit_train.yaml 中的 image / namespace / PVC / nodeSelector

# 3. 训练 YuE2-3B
kubectl apply -f k8s/submit_train.yaml
kubectl logs -f job/t2music-train-yue2 -n <namespace>

# 4. 若需跑 ACE-Step，复制 YAML 并改 command: train-acestep
```

### 3. 下载数据集

```bash
# 在挂载了足够磁盘的节点（或 K8s Job 中挂 PV）
T2MUSIC_DATA_ROOT=/data/t2music bash scripts/download_datasets.sh musiccaps --sample 100
T2MUSIC_DATA_ROOT=/data/t2music bash scripts/download_datasets.sh fma
```

### 4. 评估

```bash
# 进入 docker 或同环境容器
docker run --gpus=1 -v /data:/data t2music:phase-a eval-fad \
    --eval_dir /data/t2music/outputs/yue2 \
    --ref_dir  /data/t2music/raw/musiccaps \
    --out      /data/t2music/eval/fad.json

docker run --gpus=1 -v /data:/data t2music:phase-a eval-clap \
    --audio_dir     /data/t2music/outputs/yue2 \
    --captions_json /data/t2music/eval/demo_captions.json \
    --out           /data/t2music/eval/clap.json
```

## 验收标准（Phase A 结束，M-B1 评审）

参考 `phase_a_exec.md`：

- [ ] Docker 镜像在 K8s 可调度
- [ ] MusicCaps 至少 100 条下载并生成数据指纹
- [ ] YuE2-3B + ACE-Step v1 推理各产出 ≥20 条 demo 音频
- [ ] 盲听评分 ≥10 人 × 30 prompts 入库
- [ ] DCAE + 渲染器训练到 10k 步不 OOM
- [ ] FAD / CLAP 评估自动化脚本可用
- [ ] 基线锚点文档完成（`docs/baseline_anchor.md`）

## 许可与合规

- 模型权重：遵守各上游 License（ACE-Step Apache 2.0 / YuE2 CC BY-NC）
- 数据：PoC 阶段仅用 CC / 公共领域；商用训练需单独采购授权
- 参考 `plan.md` §9 与 `roadmap_v0.1.md` 风险表

## 联系 / 下一步

- 详细执行计划：`phase_a_exec.md`
- 技术规划：`plan.md`
- 总体路线图：`roadmap_v0.1.md`
