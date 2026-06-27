# item-level-scoring Specification

## Purpose
TBD - created by archiving change gryphon-lite-scoring-calibration. Update Purpose after archive.
## Requirements
### Requirement: SHALL 提供 Gryphon-style item-level scorer
系统 SHALL 使用 item-level scorer 重排 generated item candidates。

#### Scenario: 重排 generated candidates
- **When** 输入 generated item candidates、user representations、item embeddings 和 SID embeddings
- **Then** 每个 candidate 获得 relevance score
- **And** candidates 可以独立于 beam likelihood 重新排序

### Requirement: SHALL 使用 generated negatives 训练 scorer
item scorer SHALL 支持 positives 和 generated/hard negatives。

#### Scenario: 训练 scorer
- **When** 输入 target positives 和 generated negative candidates
- **Then** scorer 优化 ranking 或 classification loss
- **And** 报告 validation HR/NDCG 或 loss

### Requirement: SHALL 输出 Beam-vs-scorer calibration report
评估器 SHALL 比较 beam likelihood ranking 和 item scorer ranking。

#### Scenario: 度量 ranking gap
- **When** 同时拥有 beam likelihood scores 和 item scorer scores
- **Then** 报告 ranking changes、HR/NDCG difference 和 collision item separation metrics

