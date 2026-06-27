# constrained-sid-generation Specification

## Purpose
TBD - created by archiving change gryphon-lite-scoring-calibration. Update Purpose after archive.
## Requirements
### Requirement: SHALL 训练 SID Generator
系统 SHALL 训练轻量 generator，根据用户历史预测 next-item Semantic ID tokens。

#### Scenario: teacher forcing 训练 generator
- **When** 输入转换为 SID token sequences 的用户历史
- **Then** generator 优化 next SID token prediction
- **And** 报告有限 training loss

### Requirement: SHALL 支持 Trie Constrained Beam Search
generator SHALL 支持基于 catalog SID trie 的 constrained decoding。

#### Scenario: 生成合法 item candidates
- **When** 使用训练好的 generator 和 SID trie 进行 beam search
- **Then** 生成的 SID 序列属于 catalog trie
- **And** decoded candidates 能映射到 item ids

### Requirement: SHALL 报告生成合法性指标
系统 SHALL 报告生成候选的合法性和多样性。

#### Scenario: 评估 generated beams
- **When** 输入 top-k generated candidates
- **Then** 报告 valid SID rate、valid item rate、duplicate rate 和 beam diversity

