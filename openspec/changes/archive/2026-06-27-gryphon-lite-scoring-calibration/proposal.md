## 为什么需要这个变更

MiniTwoRec 不再适合继续走高成本 MiniOneRec 风格 QLoRA / RL 复现。更强、更低成本的路线是研究 Semantic ID 生成式推荐的工业失败点：

- beam likelihood 排的是 token sequence，不一定等价于 item relevance。
- 多个 item 可能 collapse 到同一个 Semantic ID，导致分数相同或校准不良。
- constrained decoding 可以保证合法 item，但不保证排序质量。

Gryphon-lite 让生成器负责 candidate generation，再用 item-level scorer 做 relevance calibration。Latte-style latent token 仅作为小消融，用于分析 decoding-tree 表达力。

## 改动内容

- 增加或标准化 KMeans / RQ-KMeans Semantic ID 构造。
- 增加小型 SID generator 和 trie constrained beam search。
- 增加 generated SID 到 item candidates 的 grounding。
- 增加 Gryphon-style item-level scoring head。
- 增加 collision-aware 和 beam-likelihood baselines。
- 增加 Latte-style latent token 消融。
- 增加合法性、collision、calibration、diversity、latency 和推荐指标。

## 不做什么

- 不做 Qwen QLoRA SFT。
- 不做 GRPO / RL 训练。
- 不做完整 MiniOneRec 复现。
- 不做多卡大规模实验。
- 不把 differentiable SID learning 作为主路线。

## 验收标准

- SID generator 能通过 constrained decoding 生成合法 item candidates。
- beam likelihood ranking 和 item-level scorer ranking 能在同一候选集上对比。
- collision case 能被统计，并能由 item scorer 重新区分排序。
- 结果包含 HR/NDCG/Recall、valid SID/item rate、duplicate rate、collision rate、beam diversity、ranking gap 和 latency。
