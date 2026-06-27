# semantic-id-tokenization Specification

## Purpose
TBD - created by archiving change gryphon-lite-scoring-calibration. Update Purpose after archive.
## Requirements
### Requirement: SHALL 支持多种 Semantic ID 构造方法
系统 SHALL 支持 random、category-aware、KMeans 和 RQ-KMeans Semantic ID 构造。

#### Scenario: 构造 item Semantic IDs
- **When** 输入 item ids、item text/category features 和指定 SID 方法
- **Then** 每个 item 获得一个 token sequence SID
- **And** 导出 item-to-SID 和 SID-to-item mappings

### Requirement: SHALL 报告 SID 质量指标
系统 SHALL 在 generator 训练前报告 SID 质量。

#### Scenario: 审计 SID tokenizer
- **When** 输入 item-to-SID mappings
- **Then** 报告 collision rate、code utilization、category purity 和 collision group 统计

