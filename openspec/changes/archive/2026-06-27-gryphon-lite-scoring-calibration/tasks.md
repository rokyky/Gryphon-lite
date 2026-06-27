## 1. Semantic ID Tokenization

- [x] 1.1 增加或标准化 random、category-aware、KMeans、RQ-KMeans SID builders。
- [x] 1.2 导出 item-to-SID 和 SID-to-item mappings。
- [x] 1.3 跟踪 SID collision groups 和 prefix metadata。
- [x] 1.4 增加 SID quality metrics：collision rate、code utilization、category purity。

## 2. SID Generator 与 Decoding

- [x] 2.1 增加小型 Transformer / SASRec-style SID generator。
- [x] 2.2 增加 next SID token teacher-forced training objective。
- [x] 2.3 增加 trie constrained beam search。
- [x] 2.4 增加 seen-item 和 duplicate filtering。
- [x] 2.5 增加 valid SID/item rate 和 beam diversity metrics。

## 3. Gryphon-style Item Scoring

- [x] 3.1 增加 generated SID 到 item candidate 的 grounding。
- [x] 3.2 增加 dot-product item scorer。
- [x] 3.3 增加 MLP item scorer，输入 user/item/SID features。
- [x] 3.4 使用 positives 和 generated/hard negatives 训练 scorer。
- [x] 3.5 对比 beam likelihood ranking 与 item-level scorer ranking。

## 4. Latte 消融

- [x] 4.1 增加 SID 生成前的 optional latent token。
- [x] 4.2 增加 latent token count 配置。
- [x] 4.3 报告 collision separation 和 hard-negative ranking 变化。

## 5. 评估与文档

- [x] 5.1 增加 Popularity、ItemCF、SASRec、TiSASRec、random SID generator baselines。
- [x] 5.2 增加 HR/NDCG/Recall。
- [x] 5.3 增加 calibration、collision、diversity、long-tail、latency 报告。
- [x] 5.4 增加 Beauty-scale 低成本实验 quickstart。
- [x] 5.5 更新 README，明确 Gryphon-lite 定位和 non-goals。
