# latte-ablation Specification

## Purpose
TBD - created by archiving change gryphon-lite-scoring-calibration. Update Purpose after archive.
## Requirements
### Requirement: SHALL 支持可选 Latte-style latent token
SID generator SHALL 支持在 SID 生成前加入可选 latent token。

#### Scenario: 开启 latent token 消融
- **When** 配置启用 latent tokens
- **Then** generator training 和 decoding 使用 latent-conditioned generation
- **And** 结果可与 vanilla generator 对比

### Requirement: SHALL 报告 Latent-token 消融指标
系统 SHALL 报告 latent token 是否改善 collision 和 hard-negative ranking。

#### Scenario: 比较 vanilla 和 latent-token generator
- **When** 输入 vanilla 和 latent-token 实验结果
- **Then** 报告 HR/NDCG、collision separation、valid item rate 和 latency 差异

