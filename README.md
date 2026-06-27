# Gryphon-lite：Semantic ID 生成式推荐与 Item-Level 评分校准

Gryphon-lite 是一个低成本、模块化的 **Semantic ID (SID) 生成式推荐**框架，核心解决生成式推荐的两个工业落地问题：

1. **beam likelihood 不等于 item relevance**：token 序列概率高，不代表 item 真正相关
2. **SID collision**：多个 item 映射到同一个 Semantic ID，beam 分数无法区分

方案：小型 SID generator 负责候选召回，轻量 item-level scorer 做相关性校准。**不做 Qwen QLoRA，不做 GRPO/RL**。

## 架构

```
item 元数据 (title, category)
    │
    ▼
文本 Embedding ──→ SID Builder ──→ SID Mappings (item_to_sid, sid_to_items)
    │                                   │
    ▼                                   ▼
用户历史 ──→ SID Generator ──→ Trie-Constrained Beam Search
    │                                   │
    ▼                                   ▼
Item Grounding ──→ 候选 Items ──→ Item-Level Scorer（重排）
                                       │
                                       ▼
                                  最终排序
```

## 项目定位

Gryphon-lite 是三项目推荐研究矩阵里的生成式推荐项目：

| 项目 | 角色 |
|---|---|
| RoTE-TimeRec | 时间建模、full-ranking 评估、评测可信度 |
| MiniMind-IntentRec | LLM / MiniMind 蒸馏用户会话意图 |
| Gryphon-lite | SID 生成式推荐与 item-level scoring 校准 |

## 核心模块

| 模块 | 文件 | 说明 |
|---|---|---|
| **SID Builder** | `src/sid_builder.py` | 将 item 映射为 Semantic ID token 序列（Random / Category / KMeans / RQ-KMeans） |
| **SID Mapper** | `src/sid_mapper.py` | 导出/加载 SID 映射；构建前缀 trie 用于受控解码 |
| **SID Metadata** | `src/sid_metadata.py` | 碰撞组、前缀统计、编码利用率追踪 |
| **SID Quality** | `src/sid_quality.py` | collision rate、code utilization、category purity 等质量指标 |
| **SID Generator** | `src/sid_generator.py` | 小型 Transformer decoder（2-4 层），从用户历史预测 next-item SID |
| **Constrained Decoder** | `src/trie_constrained_decoder.py` | Trie 约束 beam search，保生成 SID 一定在 catalog 内 |
| **Item Grounding** | `src/item_grounding.py` | 生成 SID → 候选 item 映射，含碰撞消解 |
| **Item Scorer** | `src/item_scorer.py` | Dot-product / MLP scorer，用于候选重排 |
| **Evaluation** | `src/eval_metrics.py`、`src/eval_report.py` | HR/NDCG/Recall；校准、碰撞、多样性、延迟报告 |

## Baselines

- **Popularity**：推荐全局最热 item
- **ItemCF**：简单 item-item 共现协同过滤
- **SASRec**：单头 Transformer 序列推荐
- **Random SID**：随机 SID 生成 + catalog 映射
- **SIDGen + Beam**：训练 SID generator，按 beam likelihood 排序
- **SIDGen + Scorer**：训练 SID generator + item-level scorer 重排

## 快速开始

### 环境

```bash
pip install torch numpy pandas scikit-learn scipy tqdm pyyaml
```

### 端到端流程

```bash
# 使用合成数据完整跑通（CPU 约 5 分钟）
bash scripts/quickstart.sh

# 使用真实数据
bash scripts/quickstart.sh --data-path data/Industrial_and_Scientific
```

### 分步执行

```bash
# 1. 构建 SID 映射
python -c "
from src.sid_builder import RandomSIDBuilder
from src.sid_mapper import export_mappings
item_ids = list(range(1000))
builder = RandomSIDBuilder(num_sid_tokens=3, vocab_size_per_token=256, seed=42)
item_to_sid, sid_to_items = builder.build(item_ids)
export_mappings(item_to_sid, sid_to_items, 'data/sid_mappings.json')
print(f'Built {len(sid_to_items)} unique SIDs')
"

# 2. 训练 SID generator
python scripts/train_sid_generator.py \
    --train_path data/train.csv \
    --index_path data/indices.json \
    --output_dir checkpoints/sid_generator \
    --epochs 50 --batch_size 64 --lr 1e-3

# 3. 训练 item-level scorer
python scripts/train_item_scorer.py \
    --train_path data/train.csv \
    --index_path data/indices.json \
    --sid_generator_ckpt checkpoints/sid_generator/best_model.pt \
    --output_dir checkpoints/item_scorer \
    --epochs 30 --batch_size 32

# 4. 对比排序方式
python scripts/compare_ranking.py \
    --test_path data/test.csv \
    --index_path data/indices.json \
    --sid_generator_ckpt checkpoints/sid_generator/best_model.pt \
    --scorer_ckpt checkpoints/item_scorer/best_model.pt

# 5. 运行 baselines
python scripts/run_baselines.py \
    --train_path data/train.csv \
    --test_path data/test.csv \
    --index_path data/indices.json

# 6. Latte 消融
python scripts/run_latte_ablation.py \
    --train_path data/train.csv \
    --test_path data/test.csv \
    --index_path data/indices.json \
    --output_dir ablation_results
```

## 评估体系

### 推荐指标

- **HR@K**（Hit Rate）：K 截断命中率
- **NDCG@K**（Normalized Discounted Cumulative Gain）：位置感知排序质量
- **Recall@K**：K 截断召回率

### 生成合法性指标

- **Valid SID Rate**：生成 SID 在 catalog trie 中的比例
- **Valid Item Rate**：生成 SID 能映射到真实 item 的比例
- **Duplicate Rate**：beam 输出中重复 item 的比例
- **Beam Diversity**：beam 中唯一 item 数 / beam size

### 校准指标

- **Spearman/Pearson 相关系数**：beam likelihood 与 scorer relevance 之间的相关性
- **Ranking gap**：beam 排序与 scorer 排序之间的位置偏移
- **Collision separation**：scorer 能否区分共享同一 SID 的不同 item

### 延迟指标

- **Decoding latency (ms)**：trie 约束 beam search 耗时
- **Rerank latency (ms)**：item-level scorer 前向耗时

## 当前边界与必须补的实验

当前代码层面已经实现 SID builder、SID trie、constrained beam search、item grounding、item-level scorer、baseline 与 report 模块。真正的风险在于它是 Gryphon-lite，而不是完整 Gryphon 论文复现；需要用实验说明 item-level scorer 相比 beam likelihood 的必要性。

### 已解决的代码级风险

- Random / Category-aware / KMeans / RQ-KMeans SID builders 均可构建 item-to-SID 和 SID-to-item 映射。
- SID trie 约束 beam search 能保证生成 token prefix 来自 catalog。
- Item grounding 支持 SID collision group，并能用 popularity / recency / embedding similarity 做碰撞消解。
- Dot-product / MLP item scorer 能独立于 beam likelihood 重排候选。
- Evaluation report 覆盖 calibration、collision、diversity、long-tail 和 latency。

### 当前实验硬伤

- 需要真实或 Beauty-scale 数据跑通 SID 构建、generator 训练、scorer 训练、baseline 对比和 Latte 消融。
- 必须报告 SID collision rate、collision group size、valid SID/item rate、duplicate rate 和 beam diversity。
- 必须展示 `SIDGen + Scorer` 相比 `SIDGen + Beam` 的排序收益，否则 item-level scorer 的动机不够硬。
- `run_baselines.py` 里的 SASRec / TiSASRec baseline 如果只是轻量实现，面试时要明确这是低成本对照，不是强工业 baseline。
- 必须固定 SID 构造方法、seed、beam size 和 candidate budget，否则不同方法不可比。

### 面试叙事边界

推荐表述：这是一个低成本 Gryphon-style 验证项目，核心是 Semantic ID 生成候选 + item-level scorer 校准 beam likelihood 与 item relevance 的偏差。

不推荐表述：不要说完整复现 Gryphon 或大模型生成式推荐；本项目明确不做 QLoRA、GRPO/RL 和大规模 serving。

## Latte 消融

可选地在 SID 生成前加入 learnable latent token：

```
history → latent_z → sid_1 → sid_2 → ...
```

通过 `--latent_tokens 4` 开启，对比 vanilla generator，观察 collision separation 和 hard-negative ranking 的变化。**仅作为消融，不作为主贡献。**

## 结果（占位）

### 推荐性能（Beauty-scale 合成数据）

| 方法 | HR@10 | NDCG@10 | Recall@10 |
|------|-------|---------|-----------|
| Popularity | _ | _ | _ |
| ItemCF | _ | _ | _ |
| SASRec | _ | _ | _ |
| Random SID | _ | _ | _ |
| SIDGen + Beam | _ | _ | _ |
| **SIDGen + Scorer** | _ | _ | _ |

### 生成质量

| 指标 | 值 |
|------|-----|
| Valid SID Rate | _ |
| Valid Item Rate | _ |
| Duplicate Rate | _ |
| Beam Diversity | _ |
| SID Collision Rate | _ |

### 延迟

| 阶段 | 平均延迟 | P95 延迟 |
|------|---------|---------|
| Trie-Constrained Decoding | _ | _ |
| Item Grounding | _ | _ |
| Item Scorer Rerank | _ | _ |

## 不做什么

本项目明确排除以下内容：

- **大模型 SFT**：不做 QLoRA、LoRA 或 7B+ 模型全参微调
- **强化学习**：不做 GRPO、PPO 或 policy gradient
- **复杂用户建模**：不做长期用户画像、跨 session 建模
- **全量候选检索**：使用 beam search（10-50 个候选），不做 full-catalog 检索
- **在线 serving**：不做 Redis、REST API、生产部署
- **多模态输入**：仅用文本 embedding，不做图片/音频/视频
- **分布式训练**：仅单 GPU/CPU 训练

## 算力估算与实验建议

### 最低配置

单卡 **RTX 4090（24GB）** 完全足够。KMeans / RQ-KMeans 有时更吃 CPU/RAM，GPU 不是唯一瓶颈。A100 主要是节省训练时间和调大 batch size，不是必须。

### 资源估算

| 版本 | 实验范围 | 4090 单卡 | A100 单卡 |
|------|---------|----------|----------|
| 最小闭环 | Beauty × Random SID + KMeans SID，小 SID generator，beam + scorer | 8–18 h | 6–14 h |
| **可投递可信版** | Beauty + Sports × Random/Category/KMeans/RQ-KMeans SID，beam vs scorer，collision 分析 | **30–70 h** | **20–50 h** |
| 完整实验 | 3 数据集 × 3 seeds，多 SID 长度、多 beam size、Latte ablation、latency sweep | 80–160 h | 55–120 h |

### 最容易烧钱的地方

```
beam size sweep（5/10/20 够用，不用测 50+）
SID length sweep（固定 3 tokens）
RQ-KMeans 多配置（先跑 1 组）
3 seeds 全 SID variant（只主结果补 seed）
Latte 多配置（只做 1 组对比）
```

### 必须跑的实验

```
Random SID（底线）
Category-aware SID
KMeans SID
RQ-KMeans SID
SIDGen + beam likelihood ranking（基线）
SIDGen + item-level scorer（主方法）
valid SID rate / collision rate / beam diversity / latency
long-tail vs head item 切片
```

### 可以砍的实验

- 大量 beam size sweep → 固定 5/10/20
- 大量 SID length sweep → 固定 3
- Latte 多配置 → 只做 1 组 vanilla vs latent
- 多 seed 全 SID variant → 主结果最好的 2 种 SID 补 3 seeds

### 建议跑法

1. **Beauty 小闭环**（8–18 h）：确认 SID 构建、generator 训练、scorer 训练全通
2. **补 SID 消融 + Sports**（22–52 h）：4 种 SID 方案对比、scorer calibration、beam vs scorer 排序差距
3. **Latte + 最终报告**（2–5 h）：一小消融即可，主成果是 scorer 校准

## 文件结构

```
src/
    sid_builder.py              # SID 构造策略（Random / Category / KMeans / RQ-KMeans）
    sid_mapper.py               # SID 映射导入导出、前缀 trie
    sid_metadata.py             # 碰撞组、前缀统计、编码利用率
    sid_quality.py              # SID 质量指标 + 生成合法性指标
    sid_generator.py            # Transformer decoder SID 生成器 + latent token
    trie_constrained_decoder.py # Trie 约束 beam search
    item_grounding.py           # SID → item 候选映射 + 碰撞消解
    item_scorer.py              # Dot-product / MLP item scorer
    eval_metrics.py             # HR@K / NDCG@K / Recall@K
    eval_report.py              # 校准、碰撞、多样性、延迟报告
scripts/
    train_sid_generator.py      # Teacher-forced SID generator 训练
    train_item_scorer.py        # Item scorer 训练（BPR loss）
    compare_ranking.py          # Beam vs scorer 排序对比
    run_baselines.py            # Popularity / ItemCF / SASRec / Random SID baselines
    run_latte_ablation.py       # Vanilla vs latent token 消融
    quickstart.sh               # 端到端流程脚本
rq/
    models/                     # RQ-VAE 参考实现
    datasets.py                 # Embedding 数据集
    generate_indices.py         # 通过 RQ-VAE 生成 SID 的示例
```

## License

MIT
