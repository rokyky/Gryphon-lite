# Gryphon-lite: Semantic ID Generative Recommendation with Item-Level Scoring Calibration

Gryphon-lite is a low-cost, modular framework for **Semantic ID (SID) generative recommendation** with **item-level relevance calibration**. It demonstrates that a small SID generator plus a lightweight item-level scorer can produce competitive recommendations without expensive LLM fine-tuning or reinforcement learning.

## Architecture

```
item metadata (title, category)
    |
    v
Text Embedding Model  -->  SID Builder  -->  SID Mappings (item_to_sid, sid_to_items)
    |                                          |
    v                                          v
User History  -->  SID Generator  -->  Trie-Constrained Beam Search
    |                                          |
    v                                          v
Item Grounding  -->  Candidate Items  -->  Item-Level Scorer (rerank)
                                              |
                                              v
                                        Final Ranking
```

### Core Modules

| Module | Path | Description |
|---|---|---|
| **SID Builder** | `src/sid_builder.py` | Maps items to Semantic ID token sequences (Random, Category, KMeans, RQ-KMeans) |
| **SID Mapper** | `src/sid_mapper.py` | Export/load SID mappings; build prefix trie for constrained decoding |
| **SID Metadata** | `src/sid_metadata.py` | Collision groups, prefix statistics, code utilization tracking |
| **SID Generator** | `src/sid_generator.py` | Small Transformer decoder (2-4 layers) for next-SID prediction |
| **Constrained Decoder** | `src/trie_constrained_decoder.py` | Trie-constrained beam search ensuring only valid catalog SIDs |
| **Item Grounding** | `src/item_grounding.py` | Maps generated SIDs to candidate items; collision resolution |
| **Item Scorer** | `src/item_scorer.py` | Dot-product or MLP scorer for candidate reranking |
| **Evaluation** | `src/eval_metrics.py`, `src/eval_report.py` | HR/NDCG/Recall; calibration, collision, diversity, latency reports |

### Baselines

- **Popularity**: recommend most popular items globally
- **ItemCF**: simple item-item co-occurrence collaborative filtering
- **SASRec**: single-head Transformer for sequential recommendation
- **Random SID**: generate random SIDs and map to catalog items
- **SIDGen + Beam**: trained SID generator ranked by beam likelihood
- **SIDGen + Scorer**: trained SID generator + item-level scorer reranking

## Quick Start

### Prerequisites

```bash
pip install torch numpy pandas scikit-learn scipy tqdm
```

### End-to-End Pipeline

```bash
# Full quickstart with synthetic data (~5 min on CPU)
bash scripts/quickstart.sh

# Or with real data
bash scripts/quickstart.sh --data-path data/Industrial_and_Scientific
```

### Step-by-Step

```bash
# 1. Build SID mappings
python -c "
from src.sid_builder import RandomSIDBuilder
from src.sid_mapper import export_mappings
item_ids = list(range(1000))
builder = RandomSIDBuilder(num_sid_tokens=3, vocab_size_per_token=256, seed=42)
item_to_sid, sid_to_items = builder.build(item_ids)
export_mappings(item_to_sid, sid_to_items, 'data/sid_mappings.json')
"

# 2. Train SID generator
python scripts/train_sid_generator.py \
    --train_path data/train.csv \
    --index_path data/indices.json \
    --output_dir checkpoints/sid_generator \
    --epochs 50 --batch_size 64 --lr 1e-3

# 3. Train item-level scorer
python scripts/train_item_scorer.py \
    --train_path data/train.csv \
    --index_path data/indices.json \
    --sid_generator_ckpt checkpoints/sid_generator/best_model.pt \
    --output_dir checkpoints/item_scorer \
    --epochs 30 --batch_size 32

# 4. Compare ranking methods
python scripts/compare_ranking.py \
    --test_path data/test.csv \
    --index_path data/indices.json \
    --sid_generator_ckpt checkpoints/sid_generator/best_model.pt \
    --scorer_ckpt checkpoints/item_scorer/best_model.pt

# 5. Run baselines
python scripts/run_baselines.py \
    --train_path data/train.csv \
    --test_path data/test.csv \
    --index_path data/indices.json
```

## Evaluation

### Recommendation Metrics

- **HR@K**: Hit Rate at cutoff K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Recall@K**: Recall at cutoff K

### Generation Quality Metrics

- **Valid SID Rate**: fraction of generated SIDs that exist in the catalog
- **Valid Item Rate**: fraction of generated SIDs that map to real items
- **Duplicate Rate**: duplicate items in generated candidates
- **Beam Diversity**: unique / total items in beam output

### Calibration Metrics

- **Spearman/Pearson correlation**: beam likelihood vs. scorer relevance
- **Ranking gap**: position shift between beam and scorer rankings
- **Collision separation**: can the scorer distinguish items sharing the same SID?

### Latency Metrics

- **Decoding latency** (ms): trie-constrained beam search
- **Rerank latency** (ms): item-level scorer forward pass

## Non-Goals

This project explicitly does **NOT** include:

- **Large language model fine-tuning**: No QLoRA, LoRA, or full-parameter SFT on 7B+ models
- **Reinforcement learning**: No GRPO, PPO, or policy gradient methods
- **Complex user modeling**: No long-term user profiles, cross-session modeling, or user ID embeddings beyond history aggregation
- **Exhaustive candidate retrieval**: Uses beam search (typically 10-50 candidates), not full-catalog retrieval
- **Online serving infrastructure**: No Redis, no REST API, no production deployment
- **Multi-modal inputs**: Text embeddings only; no images, audio, or video features
- **Distributed training**: Single-GPU or CPU training only

## Project Context

Gryphon-lite is the generative recommendation project in a three-project matrix:

| Project | Role |
|---|---|
| **RoTE-TimeRec** | Temporal modeling, full-ranking evaluation, benchmark trustworthiness |
| **MiniMind-IntentRec** | LLM / MiniMind distillation of user session intent |
| **Gryphon-lite** | SID generative recommendation with item-level scoring calibration |

## File Structure

```
src/
    sid_builder.py           # SID construction strategies
    sid_mapper.py            # Mapping export/load and SID trie
    sid_metadata.py          # Collision/prefix/utilization metadata
    sid_quality.py           # SID quality and generation metrics
    sid_generator.py         # Transformer decoder SID generator + latent tokens
    trie_constrained_decoder.py  # Trie-constrained beam search
    item_grounding.py        # SID-to-item grounding with collision resolution
    item_scorer.py           # Dot-product and MLP item scorers
    eval_metrics.py          # HR@K, NDCG@K, Recall@K
    eval_report.py           # Calibration, collision, diversity, latency reports
scripts/
    train_sid_generator.py   # Teacher-forced SID generator training
    train_item_scorer.py     # Item scorer training with BPR loss
    compare_ranking.py       # Beam vs scorer ranking comparison
    run_baselines.py         # Popularity, ItemCF, SASRec, Random SID baselines
    run_latte_ablation.py    # Vanilla vs latent token ablation
    quickstart.sh            # End-to-end pipeline script
rq/
    models/                  # RQ-VAE reference implementation
    datasets.py              # Embedding dataset
    generate_indices.py      # Example SID generation via RQ-VAE
```

## License

MIT
