## 总体设计

Gryphon-lite 将 MiniTwoRec 重构为低成本 Semantic ID generative retrieval 项目。它保留“生成 item Semantic ID”的核心思想，但用小型生成器和 item-level scoring calibration 替代大模型 SFT / RL。

## Pipeline

```text
item text/category
-> text embedding
-> KMeans/RQ-KMeans Semantic ID
-> SID generator
-> trie constrained beam search
-> generated SID -> item candidates
-> item-level scorer rerank
```

## Semantic ID 设计

支持以下 SID baselines：

- random ID
- category-aware ID
- KMeans SID
- RQ-KMeans SID

SID metadata 必须包含：

- item id
- SID tokens
- SID prefix
- collision group id

## Generator 设计

使用小型 Transformer / SASRec-style decoder，从用户历史预测 next SID tokens。

生成器必须支持：

- teacher-forced training
- beam search
- trie constrained decoding
- seen-item filtering
- duplicate filtering

## Gryphon-style Item Scorer

item scorer 对生成候选重新打分：

```text
score = f(user_embedding, item_embedding, sid_embedding, optional features)
```

至少支持 dot-product 和 MLP 两种模式。训练使用 next-item positives 和 generated/hard negatives。

## Latte 消融

可选地在 SID 生成前加入 learnable latent token：

```text
history -> latent_z -> sid_1 -> sid_2 -> ...
```

该部分只作为消融，不作为主贡献。主要观察 latent token 是否改善 collision separation 或 hard-negative ranking。

## 评估设计

Baselines：

- Popularity
- ItemCF
- SASRec
- TiSASRec
- random SID generator
- semantic SID generator + beam likelihood ranking
- collision-resolved ranking
- Gryphon-lite item-level scorer
- Gryphon-lite + Latte latent-token ablation

指标：

- HR/NDCG/Recall
- valid SID rate
- valid item rate
- duplicate rate
- SID collision rate
- beam diversity
- beam-likelihood vs item-scorer ranking gap
- collision item separation
- long-tail hit rate
- decoding latency 和 rerank latency

## 风险

- generator overall 指标可能不如 SASRec。只要能分析 Semantic ID 在哪些切片有效或失败，项目仍成立。
- SID 构造方式会强烈影响结果，因此必须包含 random/category/KMeans/RQ-KMeans 对比。
- item scorer 可能只学 popularity，因此需要报告 long-tail 和 collision slices。
