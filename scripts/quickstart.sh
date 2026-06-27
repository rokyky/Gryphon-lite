#!/usr/bin/env bash
# Gryphon-lite Quickstart: end-to-end scoring calibration pipeline (Task 5.4)
#
# Steps:
#   1. Build SID mappings using RandomSIDBuilder
#   2. Export SID mappings
#   3. Train SID generator (small Transformer decoder)
#   4. Run trie-constrained beam search decoding
#   5. Train item-level scorer (MLP)
#   6. Compare beam likelihood vs scorer ranking
#   7. Generate evaluation report
#
# Usage:
#   bash scripts/quickstart.sh [--synthetic] [--data-path data/Industrial_and_Scientific]
#
# This script uses small hyperparameters for quick experimentation.
# On a CPU, the full pipeline takes approximately 5-10 minutes.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# Defaults
DATA_PATH="${DATA_PATH:-data/Industrial_and_Scientific}"
SYNTHETIC="${SYNTHETIC:-false}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"

echo "============================================"
echo " Gryphon-lite Quickstart"
echo "============================================"
echo "Project dir: $PROJECT_DIR"
echo "Data path:   $DATA_PATH"
echo "Epochs:      $EPOCHS"
echo "Batch size:  $BATCH_SIZE"
echo "Device:      $DEVICE"
echo "============================================"

# Step 0: Generate synthetic data if requested or if real data missing
if [ "$SYNTHETIC" = "true" ] || [ ! -f "$DATA_PATH/train/sequences.csv" ]; then
    echo ""
    echo "[Step 0] Generating synthetic data..."
    mkdir -p "$DATA_PATH/train" "$DATA_PATH/valid" "$DATA_PATH/test"

    python -c "
import json, random, numpy as np
random.seed(42)
np.random.seed(42)

num_items = 1000
num_users = 200
num_sid_tokens = 3
vocab_per_token = 256

# Generate random item embeddings
embeddings = np.random.randn(num_items, 64).astype(np.float32)
np.save('$DATA_PATH/embeddings.npy', embeddings)

# Generate random SID assignments
item_to_sid = {}
for i in range(num_items):
    sid = tuple(random.randint(0, vocab_per_token - 1) for _ in range(num_sid_tokens))
    item_to_sid[str(i)] = list(sid)

with open('$DATA_PATH/indices.json', 'w') as f:
    json.dump(item_to_sid, f)
print(f'Saved indices.json with {num_items} items')

# Generate user sequences
def gen_sequences(num_seqs, prefix):
    import csv, os
    os.makedirs(os.path.dirname('$DATA_PATH/' + prefix), exist_ok=True)
    with open('$DATA_PATH/' + prefix + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['history_item_id', 'item_id'])
        for _ in range(num_seqs):
            seq_len = random.randint(3, 10)
            history = [str(random.randint(0, num_items - 1)) for _ in range(seq_len)]
            target = str(random.randint(0, num_items - 1))
            writer.writerow([history, target])
    print(f'Generated {num_seqs} sequences at $DATA_PATH/{prefix}.csv')

gen_sequences(500, 'train/sequences')
gen_sequences(100, 'valid/sequences')
gen_sequences(200, 'test/sequences')
print('Synthetic data generation complete.')
"
fi

# Step 1: Build SID mappings (using RandomSIDBuilder)
echo ""
echo "[Step 1] Building SID mappings..."
python -c "
import json, sys
sys.path.insert(0, '$PROJECT_DIR')
from src.sid_builder import RandomSIDBuilder
from src.sid_mapper import export_mappings

with open('$DATA_PATH/indices.json') as f:
    index = json.load(f)

item_ids = list(index.keys())
builder = RandomSIDBuilder(num_sid_tokens=3, vocab_size_per_token=256, seed=42)
item_to_sid, sid_to_items = builder.build(item_ids)

export_path = '$DATA_PATH/sid_mappings.json'
export_mappings(item_to_sid, sid_to_items, export_path)
print(f'SID mappings exported to {export_path}')
print(f'Unique SIDs: {len(sid_to_items)}, Collisions: {len(item_ids) - len(sid_to_items)}')
"

# Step 2: Train SID generator
echo ""
echo "[Step 2] Training SID generator..."
mkdir -p checkpoints/sid_generator

python scripts/train_sid_generator.py \
    --train_path "$DATA_PATH/train/sequences.csv" \
    --valid_path "$DATA_PATH/valid/sequences.csv" \
    --index_path "$DATA_PATH/indices.json" \
    --output_dir checkpoints/sid_generator \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --hidden_dim 64 \
    --num_layers 2 \
    --num_heads 2 \
    --device $DEVICE \
    --seed 42

echo "SID generator training complete."

# Step 3: Train item-level scorer
echo ""
echo "[Step 3] Training item-level scorer..."
mkdir -p checkpoints/item_scorer

python scripts/train_item_scorer.py \
    --train_path "$DATA_PATH/train/sequences.csv" \
    --valid_path "$DATA_PATH/valid/sequences.csv" \
    --index_path "$DATA_PATH/indices.json" \
    --sid_generator_ckpt checkpoints/sid_generator/best_model.pt \
    --output_dir checkpoints/item_scorer \
    --scorer_type mlp \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --user_dim 64 \
    --item_dim 64 \
    --sid_dim 64 \
    --hidden_dims 128 64 \
    --device $DEVICE \
    --seed 42

echo "Item scorer training complete."

# Step 4: Compare ranking methods
echo ""
echo "[Step 4] Comparing beam likelihood vs item scorer ranking..."
mkdir -p results

python scripts/compare_ranking.py \
    --test_path "$DATA_PATH/test/sequences.csv" \
    --index_path "$DATA_PATH/indices.json" \
    --sid_generator_ckpt checkpoints/sid_generator/best_model.pt \
    --scorer_ckpt checkpoints/item_scorer/best_model.pt \
    --beam_width 20 \
    --num_samples 100 \
    --output results/ranking_comparison.json \
    --device $DEVICE

echo "Ranking comparison complete."

# Step 5: Run baselines (optional, can be slow)
echo ""
echo "[Step 5] Running baselines (optional)..."
python scripts/run_baselines.py \
    --train_path "$DATA_PATH/train/sequences.csv" \
    --test_path "$DATA_PATH/test/sequences.csv" \
    --index_path "$DATA_PATH/indices.json" \
    --sid_generator_ckpt checkpoints/sid_generator/best_model.pt \
    --output results/baselines.json \
    --ks 5 10 20 \
    --device $DEVICE

echo ""
echo "============================================"
echo " Gryphon-lite Quickstart Complete!"
echo "============================================"
echo "Results saved to:"
echo "  - SID mappings:      $DATA_PATH/sid_mappings.json"
echo "  - SID generator:     checkpoints/sid_generator/best_model.pt"
echo "  - Item scorer:       checkpoints/item_scorer/best_model.pt"
echo "  - Ranking compare:   results/ranking_comparison.json"
echo "  - Baselines:         results/baselines.json"
echo "============================================"
