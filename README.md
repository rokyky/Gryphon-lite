# MiniTwoRec — 基于 Semantic ID 的生成式推荐框架

MiniTwoRec 是首个完全开源的**生成式推荐**框架，覆盖 **SID 构造** -> **SFT** -> **推荐导向 RL（GRPO）** 完整流程。
核心思路：用 RQ-VAE 将物品编码为紧凑的语义 ID（SID），通过因果 Transformer 建模用户行为序列，
并用 GRPO 强化学习优化推荐质量。

| | |
|---|---|
| 论文 | [arXiv 2510.24431](https://arxiv.org/abs/2510.24431) |
| 代码 | [github.com/AkaliKong/MiniTwoRec](https://github.com/AkaliKong/MiniTwoRec) |
| 模型 | [HuggingFace](https://huggingface.co/kkknight/MiniTwoRec) |
| 参考 | OneRec (arXiv 2502.18965) | TIGER (arXiv 2206.03975) |

---

## 项目定位

在三个项目组合中，MiniTwoRec 承担**生成式推荐前沿**角色，与 TimeGenRec（打分式）互补对照。
两者共享 Amazon 数据集和 full-ranking 评估协议，但模型范式和训练方式完全不同。

## 三阶段流程

1. **SID 构造**：item title+description -> 文本编码器（Qwen）-> 3 层 RQ-VAE -> 离散语义 ID
2. **SFT**：用户历史序列（sid 序列）+ item 对齐 -> next SID prediction + 语言对齐目标
3. **推荐导向 RL（GRPO）**：多候选生成 -> 组内奖励归一化 -> KL 惩罚 -> 约束 beam search

## 文件结构

| 文件/目录 | 说明 |
|----------|------|
| src/ | 核心代码（sft / rl / evaluate / data 等）|
| rq/ | RQ-VAE / RQ-Kmeans SID 构造 |
| baselines/ | SASRec 等 baseline 模型 |
| ts_rec/ | TS-Rec 扩展 |
| config/ | DeepSpeed ZeRO-2 配置 |
| data/ | Amazon 数据预处理脚本 |
| scripts/ | 训练/评估/数据转换入口脚本 |
| assets/ | 图片等资源 |

---

## 数据集

| 数据集 | 用途 | 用户数 | 物品数 |
|--------|------|--------|--------|
| Amazon Industrial & Scientific | 主实验 | ~20K | ~15K |
| Amazon Office Products | 辅助验证 | ~25K | ~12K |
| Amazon Beauty / Sports（与 TimeGenRec 共享）| 兼容 | 22-35K | 12-18K |

---

## GPU 与训练配置

| 阶段 | 推荐 GPU | 显存 | 时长 | 租用价 |
|------|---------|------|------|--------|
| RQ-VAE | 1x RTX 4090 24GB | ~8-16GB | 2-3h | ~¥7 |
| RQ-VAE | 1x A100 80GB | ~8GB | 0.5-1h | ~¥8 |
| 文本编码 | 1x A100 80GB / CPU | ~16GB | 30m-1h | ~¥8 |
| SFT (QLoRA 4-bit) | 1x RTX 4090 24GB | ~20GB | 8-12h | ~¥25 |
| SFT (全精度) | **4-8x A100 80GB** | ~40-60GB | 2-4h | ~¥60-100 |
| RL (GRPO) | 4-8x A100 80GB | ~60-80GB | 2-4h | ~¥60-100 |
| 评估 | 1x RTX 4090 / A100 | ~8GB | 30m | ~¥5 |

**推荐方案：** 开发和验证阶段用 4090 + QLoRA（~¥25/次），最终结果用 8xA100 复现（~¥60-100/次）。

### 超参数

| 阶段 | 参数 |
|------|------|
| RQ-VAE | lr=1e-3, epochs=10000, batch=20480, 3-level, codebook=512 |
| SFT | batch=1024, micro_batch=16, ZeRO-2, bf16, AdamW |
| RL | 约束 beam search, 二元正确性+排名感知奖励, KL 惩罚 |

---

## 评估（官方结果 — Amazon Industrial & Scientific）

| 模型 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|------|------|-------|--------|---------|
| SASRec | ~0.05 | ~0.09 | ~0.03 | ~0.05 |
| BERT4Rec | ~0.06 | ~0.11 | ~0.04 | ~0.06 |
| **MiniTwoRec** | **~0.08** | **~0.14** | **~0.05** | **~0.08** |

## 快速开始

```bash
pip install -r requirements.txt
bash data/amazon18_data_process.sh --dataset Industrial_and_Scientific --user_k 5 --item_k 5
bash rq/amazon_text2emb.sh --dataset Industrial_and_Scientific --plm_name qwen
bash rq/rqvae.sh
python rq/generate_indices.py
python convert_dataset.py --dataset_name Industrial_and_Scientific
bash sft.sh   # 8x A100 或 QLoRA 4090
bash rl.sh     # 可选，需要多卡
bash evaluate.sh
```

---

## 引用

```bib
@misc{MiniTwoRec,
    title={MiniTwoRec: An Open-Source Framework for Scaling Generative Recommendation},
    author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and ...},
    year={2025}, eprint={2510.24431}, archivePrefix={arXiv},
}
```

## License: Apache 2.0