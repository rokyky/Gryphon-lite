## Why MiniTwoRec

Generative recommendation with Semantic ID. RQ-VAE encodes items into discrete semantic codes,
causal Transformer generates next-item codes, SFT + DPO optimize recommendation quality.

Code exists in 5.time-aware-seqrec research line (rq_vae.py 410L, semantic_id.py 211L, genrec.py 244L, gen_eval.py 311L).
This project ports and productionizes it.

## What Changes
- Port RQ-VAE / Semantic ID / GenRec models
- Amazon text metadata preprocessing (title, category, description -> Semantic ID input)
- Generative eval protocol: reconstruction quality + full-ranking recall + codebook utilization
- Beam search decoding and candidate generation
- SFT + recommendation-oriented DPO

## Capabilities
- semantic-id: RQ-VAE training and codebook construction
- gen-retrieval: Causal Transformer generative retrieval
- gen-eval: reconstruction quality, codebook utilization, generative recall
- data-prep: Amazon text metadata preprocessing
- sft-rl: SFT + DPO training

## Impact
New project. Models ported from 5.time-aware-seqrec, datasets and eval independently built.