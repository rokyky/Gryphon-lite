## ADDED Requirements

### Requirement: RQ-VAE codebook training
RQ-VAE must encode item embeddings into 3-level discrete codes with residual quantization.
Codebook size=512 per level, EMA update to prevent collapse.

### Requirement: Semantic ID construction
Each item gets a hierarchical code [l1,l2,l3] from RQ-VAE quantization.

### Requirement: GenRec causal Transformer
Causal Transformer predicts next item code sequence. Uses code flattening.

### Requirement: Beam search decoding
Top-K candidate generation via beam search with constraint decoding (exclude seen items).

### Requirement: Generative eval
Eval reconstruction quality (item recall from reconstructed emb) and recommendation quality (full-ranking recall).

### Requirement: Amazon text preprocessing
Extract title, category, description from Amazon metadata. Build text embeddings via Sentence-BERT.