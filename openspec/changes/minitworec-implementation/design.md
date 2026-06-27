## Context

RQ-VAE: 410 lines of working code
Semantic ID: 211 lines of working code
GenRec: 244 lines of working code
GenEval: 311 lines of working code
All in 5.time-aware-seqrec/src/models/ and src/utils/

## Goals / Non-Goals

Goals:
- Port models to independent project
- Amazon text metadata -> Semantic ID pipeline
- Full-ranking eval for generative models
- SFT + DPO training

Non-Goals:
- Production serving
- 100M+ param models
- External framework dependencies

## Design Decisions

D1: Semantic ID from item text embedding (Sentence-BERT on title+category)
D2: RQ-VAE with 3-level residual quantization, codebook size 512 per level
D3: GenRec uses RoPE + GQA + RMSNorm (Mini-DeepLLM architecture components)
D4: Eval in two layers: reconstruction quality (item recall) + recommendation (full-ranking)

## Dataset

Amazon Beauty/Sports text metadata (title, category, description) as Semantic ID input.
Shares interaction data with TimeGenRec.

| Dataset | Purpose | Text Features |
|---------|---------|---------------|
| Amazon Beauty | Primary | title + category + description |
| Amazon Sports | Secondary | title + category + description |
| Yelp | Optional | title + category + reviews |

## Risks
- Codebook collapse: RQ-VAE may use only a few codes -> monitor utilization
- Text quality: Amazon metadata noise may degrade Semantic ID -> filter low-quality entries
- Train stability: GenRec causal Transformer on long code sequences -> gradient clipping