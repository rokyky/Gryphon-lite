#!/usr/bin/env python3
"""
Run baselines for Gryphon-lite evaluation (Task 5.1).

Baselines:
    1. Popularity: recommend most popular items globally
    2. ItemCF: simple item-item co-occurrence collaborative filtering
    3. SASRec: simple Transformer-based sequential recommendation
    4. Random SID generator: generate random SIDs (no training)
    5. Semantic SID generator + beam likelihood ranking (from trained model)

Outputs a comparison table with HR/NDCG/Recall metrics.

Usage:
    python scripts/run_baselines.py \
        --train_path data/train.csv \
        --valid_path data/valid.csv \
        --test_path data/test.csv \
        --index_path data/indices.json \
        --output results/baselines.json
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval_metrics import hr_at_k, ndcg_at_k, recall_at_k
from src.sid_generator import SIDGenerator, SIDGeneratorConfig
from src.sid_mapper import SIDTrie, build_sid_trie
from src.sid_builder import RandomSIDBuilder
from src.trie_constrained_decoder import TrieConstrainedBeamSearch, TrieBeamSearchConfig

logger = logging.getLogger(__name__)


# ===== Data Loading =====

def load_sequences(train_path, valid_path, test_path, index_path, max_history_len=50, min_seq_len=2):
    """Load interaction sequences for all baselines.

    Returns:
        train_sequences: list of (user_id, history_item_ids, next_item_id)
        valid_sequences: same structure
        test_sequences: same structure
        item_to_sid: dict of item_id -> SID tuple
        sid_to_items: reverse mapping
        all_item_ids: complete item catalog
        item_popularity: dict of item_id -> interaction count
    """
    with open(index_path, 'r') as f:
        index = json.load(f)
    item_to_sid = {iid: tuple(int(s) for s in sid_list) for iid, sid_list in index.items()}
    sid_to_items = defaultdict(list)
    for iid, sid in item_to_sid.items():
        sid_to_items[sid].append(iid)

    all_item_ids = list(item_to_sid.keys())

    import pandas as pd

    def parse_sequences(path):
        if not path or not os.path.exists(path):
            return [], Counter()
        data = pd.read_csv(path)
        sequences = []
        pop_counter = Counter()
        for idx in range(len(data)):
            row = data.iloc[idx]
            try:
                history_ids = eval(str(row.get('history_item_id', '[]')))
            except (ValueError, SyntaxError):
                history_ids = []
            target_id = str(row.get('item_id', ''))
            # Filter to known items
            history_ids = [str(h) for h in history_ids if str(h) in item_to_sid]
            target_id = str(target_id) if target_id in item_to_sid else None
            if len(history_ids) >= min_seq_len and target_id is not None:
                if len(history_ids) > max_history_len:
                    history_ids = history_ids[-max_history_len:]
                sequences.append((str(idx), history_ids, target_id))
                pop_counter[target_id] += 1
        logger.info(f"Loaded {len(sequences)} sequences from {path}")
        return sequences, pop_counter

    train_seq, train_pop = parse_sequences(train_path)
    valid_seq, valid_pop = parse_sequences(valid_path)
    test_seq, test_pop = parse_sequences(test_path)

    # Aggregate popularity across all splits
    item_popularity = train_pop + valid_pop + test_pop

    return (train_seq, valid_seq, test_seq, item_to_sid,
            dict(sid_to_items), all_item_ids, item_popularity)


# ===== Baseline 1: Popularity =====

class PopularityBaseline:
    """Recommend the most popular items globally."""

    def __init__(self, item_popularity: Dict[str, int]):
        self.ranked_items = [
            item_id for item_id, _ in
            sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        ]
        logger.info(f"Popularity baseline: top item = {self.ranked_items[0] if self.ranked_items else None}")

    def recommend(self, history_items: List[str], k: int = 20) -> List[str]:
        """Return top-k popular items, excluding those already seen."""
        seen = set(history_items)
        candidates = [i for i in self.ranked_items if i not in seen]
        return candidates[:k]


# ===== Baseline 2: ItemCF =====

class ItemCFBaseline:
    """Simple item-item co-occurrence collaborative filtering.

    Score(item | user_history) = sum_{h in history} cooccurrence(h, item)
    Co-occurrence is normalized by sqrt(popularity(h) * popularity(item)).
    """

    def __init__(self, sequences: List[Tuple], min_cooccurrence: int = 2):
        self.item_popularity: Dict[str, int] = Counter()
        self.cooccurrence: Dict[Tuple[str, str], int] = Counter()

        for _, history, target in sequences:
            self.item_popularity[target] += 1
            all_items_in_session = set(history + [target])
            items_list = list(all_items_in_session)
            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    a, b = items_list[i], items_list[j]
                    if a < b:
                        self.cooccurrence[(a, b)] += 1
                    else:
                        self.cooccurrence[(b, a)] += 1

        # Filter low co-occurrence
        self.cooccurrence = {
            pair: count for pair, count in self.cooccurrence.items()
            if count >= min_cooccurrence
        }
        logger.info(f"ItemCF: {len(self.cooccurrence)} co-occurrence pairs, "
                     f"{len(self.item_popularity)} items")

    def recommend(self, history_items: List[str], k: int = 20) -> List[str]:
        """Score all items by co-occurrence with history."""
        scores: Dict[str, float] = Counter()
        seen = set(history_items)

        for h in history_items:
            for (a, b), count in self.cooccurrence.items():
                if a == h:
                    other = b
                elif b == h:
                    other = a
                else:
                    continue
                if other in seen:
                    continue
                # Normalize by sqrt(popularity)
                pop_h = self.item_popularity.get(h, 1)
                pop_o = self.item_popularity.get(other, 1)
                scores[other] += count / math.sqrt(pop_h * pop_o + 1)

        ranked = [item for item, _ in scores.most_common(k)]
        # Fall back to popularity if not enough
        if len(ranked) < k:
            pop_items = sorted(self.item_popularity.keys(),
                               key=lambda x: self.item_popularity[x], reverse=True)
            for item in pop_items:
                if item not in seen and item not in ranked:
                    ranked.append(item)
                    if len(ranked) >= k:
                        break
        return ranked[:k]


# ===== Baseline 3: SASRec =====

class SASRecBaseline(torch.nn.Module):
    """Simple single-head Transformer for sequential recommendation.

    Uses a simplified SASRec-style architecture:
        - Item embedding
        - Positional encoding
        - 2-layer Transformer decoder (causal)
        - Output projection to item scores
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.item_embedding = torch.nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, hidden_dim)

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = torch.nn.Linear(hidden_dim, num_items + 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            item_ids: (batch, seq_len) item id sequence.

        Returns:
            torch.Tensor: (batch, seq_len, num_items+1) scores for each position.
        """
        B, L = item_ids.shape
        positions = torch.arange(L, device=item_ids.device).unsqueeze(0).expand(B, -1)

        x = self.item_embedding(item_ids) + self.pos_embedding(positions)
        causal_mask = torch.nn.TransformerDecoder.generate_square_subsequent_mask(L).to(item_ids.device)

        # SASRec uses the decoder in a self-attention only fashion (no cross-attention)
        # We pass a dummy memory (zeros) since the decoder needs it
        memory = torch.zeros_like(x)
        out = self.decoder(x, memory, tgt_mask=causal_mask)
        logits = self.output_proj(out)  # (B, L, num_items+1)
        return logits

    @torch.no_grad()
    def recommend(self, history_ids: List[str], item_to_idx: Dict[str, int],
                  idx_to_item: Dict[int, str], k: int = 20) -> List[str]:
        """Generate recommendations from history."""
        self.eval()
        device = next(self.parameters()).device

        mapped = [item_to_idx.get(i, 0) for i in history_ids]
        if len(mapped) > self.max_seq_len:
            mapped = mapped[-self.max_seq_len:]

        input_tensor = torch.tensor([mapped], dtype=torch.long, device=device)
        logits = self.forward(input_tensor)  # (1, L, V)
        last_logits = logits[0, -1, :]  # (V,)

        # Mask seen items
        seen_indices = set(mapped)
        last_logits[list(seen_indices)] = float('-inf')

        scores, indices = torch.topk(last_logits, k)
        items = [idx_to_item[int(i)] for i in indices.cpu().numpy() if int(i) in idx_to_item]
        return items


def train_sasrec(
    model: SASRecBaseline,
    sequences: List[Tuple],
    item_to_idx: Dict[str, int],
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu'),
) -> SASRecBaseline:
    """Train SASRec baseline with teacher forcing."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Build training sequences
    train_seqs = []
    for _, history, target in sequences:
        mapped = [item_to_idx.get(i, 0) for i in history]
        train_seqs.append((mapped, item_to_idx.get(target, 0)))
    logger.info(f"SASRec training: {len(train_seqs)} sequences")

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_seqs)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_seqs), batch_size):
            batch = train_seqs[i:i + batch_size]
            inputs = []
            labels = []
            for hist, target in batch:
                inputs.append(hist)
                labels.append(target)

            # Pad sequences
            max_len = max(len(s) for s in inputs)
            padded = []
            for s in inputs:
                pad_len = max_len - len(s)
                padded.append([0] * pad_len + s)

            input_tensor = torch.tensor(padded, dtype=torch.long, device=device)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits = model(input_tensor)  # (B, L, V)

            # Use last position logits for prediction
            last_logits = logits[:, -1, :]  # (B, V)
            loss = torch.nn.functional.cross_entropy(last_logits, label_tensor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"SASRec Epoch {epoch}/{epochs} | Loss: {total_loss / max(num_batches, 1):.4f}")

    return model


# ===== Baseline 4: Random SID Generator =====

class RandomSIDGeneratorBaseline:
    """Generate random SIDs and map back to items.

    No training: just draws random SID tokens uniformly.
    """

    def __init__(self, sid_to_items: Dict[Tuple[int, ...], List[str]],
                 num_tokens: int = 3, vocab_per_token: int = 256):
        self.sid_to_items = sid_to_items
        self.num_tokens = num_tokens
        self.vocab_per_token = vocab_per_token
        self.all_sids = list(sid_to_items.keys())

    def recommend(self, history_items: List[str], k: int = 20) -> List[str]:
        """Randomly sample SIDs and return their items."""
        candidates = []
        seen = set(history_items)
        # Sample random SIDs from the catalog
        sampled_sids = random.choices(self.all_sids, k=k * 3)
        for sid in sampled_sids:
            items = self.sid_to_items.get(sid, [])
            for item_id in items:
                if item_id not in seen and item_id not in candidates:
                    candidates.append(item_id)
                    if len(candidates) >= k:
                        break
            if len(candidates) >= k:
                break
        # Fallback if not enough
        while len(candidates) < k:
            sid = random.choice(self.all_sids)
            items = self.sid_to_items.get(sid, [])
            for item_id in items:
                if item_id not in candidates:
                    candidates.append(item_id)
                    break
        return candidates[:k]


# ===== Baseline 5: Trained SID Generator + Beam Search =====

class TrainedSIDGeneratorBaseline:
    """Use trained SID generator with trie-constrained beam search."""

    def __init__(self, model_ckpt: str, trie: SIDTrie,
                 beam_width: int = 20, device: torch.device = torch.device('cpu')):
        ckpt = torch.load(model_ckpt, map_location='cpu')
        cfg = ckpt.get('config', SIDGeneratorConfig())
        self.model = SIDGenerator(cfg).to(device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.trie = trie
        self.beam_width = beam_width
        self.device = device
        beam_cfg = TrieBeamSearchConfig(beam_width=beam_width)
        self.decoder = TrieConstrainedBeamSearch(trie, beam_cfg)
        self.num_sid_tokens = cfg.num_sid_tokens

    def recommend(self, history_sids: List[Tuple[int, ...]],
                  sid_to_items: Dict[Tuple[int, ...], List[str]],
                  k: int = 20) -> List[str]:
        if not history_sids:
            return []
        B, H, T = 1, len(history_sids), self.num_sid_tokens
        hist_tensor = torch.tensor([[history_sids]], dtype=torch.long, device=self.device)
        hist = hist_tensor.reshape(B, H, T)

        sequences, scores = self.decoder.search(hist, self.model, num_return=self.beam_width)

        seen = set()
        candidates = []
        for sid in sequences[0]:
            if sid in sid_to_items:
                for item_id in sid_to_items[sid]:
                    if item_id not in seen:
                        seen.add(item_id)
                        candidates.append(item_id)
                        if len(candidates) >= k:
                            break
                if len(candidates) >= k:
                    break
        return candidates[:k]


# ===== Evaluation =====

def evaluate_baseline(name: str, recommend_fn, test_sequences: List[Tuple],
                      ks: List[int], **kwargs) -> Dict:
    """Evaluate a baseline on test sequences.

    Args:
        name: baseline name.
        recommend_fn: callable that takes (history_items, k) -> list of items.
        test_sequences: list of (user_id, history, target).
        ks: list of K values.

    Returns:
        Dict with HR, NDCG, Recall for each K.
    """
    hr_results = {k: [] for k in ks}
    ndcg_results = {k: [] for k in ks}
    recall_results = {k: [] for k in ks}

    for _, history, target in tqdm(test_sequences, desc=f'{name}'):
        try:
            preds = recommend_fn(history, k=max(ks))
        except Exception:
            preds = []

        for k in ks:
            hr_results[k].append(hr_at_k(preds, [target], k))
            ndcg_results[k].append(ndcg_at_k(preds, [target], k))
            recall_results[k].append(recall_at_k(preds, [target], k))

    metrics = {}
    for k in ks:
        metrics[f'HR@{k}'] = float(np.mean(hr_results[k])) if hr_results[k] else 0.0
        metrics[f'NDCG@{k}'] = float(np.mean(ndcg_results[k])) if ndcg_results[k] else 0.0
        metrics[f'Recall@{k}'] = float(np.mean(recall_results[k])) if recall_results[k] else 0.0

    return metrics


def print_comparison_table(all_metrics: Dict[str, Dict], ks: List[int]):
    """Print formatted comparison table."""
    print('\n' + '=' * 80)
    print('Baseline Comparison Table')
    print('=' * 80)

    header = f"{'Baseline':<35}"
    for k in ks:
        header += f"{'HR@'+str(k):<10}{'NDCG@'+str(k):<10}"
    print(header)
    print('-' * 80)

    for name, metrics in all_metrics.items():
        row = f"{name:<35}"
        for k in ks:
            hr = metrics.get(f'HR@{k}', 0.0)
            ndcg = metrics.get(f'NDCG@{k}', 0.0)
            row += f"{hr:<10.4f}{ndcg:<10.4f}"
        print(row)

    print('=' * 80)


# ===== Main =====

def parse_args():
    parser = argparse.ArgumentParser(description='Run Gryphon-lite baselines')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--sid_generator_ckpt', type=str, default=None,
                        help='Path to trained SID generator (optional)')
    parser.add_argument('--output', type=str, default='results/baselines.json')
    parser.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--max_history_len', type=int, default=50)
    parser.add_argument('--num_sid_tokens', type=int, default=3)
    parser.add_argument('--vocab_per_token', type=int, default=256)
    parser.add_argument('--beam_width', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f'Using device: {device}')

    # Load data
    train_seq, valid_seq, test_seq, item_to_sid, sid_to_items, all_item_ids, item_pop = \
        load_sequences(args.train_path, args.valid_path, args.test_path,
                       args.index_path, args.max_history_len)

    if not test_seq:
        logger.error('No test sequences loaded. Aborting.')
        return

    ks = args.ks
    all_metrics = {}

    # 1. Popularity baseline
    logger.info('Running Popularity baseline...')
    pop_baseline = PopularityBaseline(item_pop)
    all_metrics['Popularity'] = evaluate_baseline(
        'Popularity', pop_baseline.recommend, test_seq, ks)

    # 2. ItemCF baseline
    logger.info('Running ItemCF baseline...')
    itemcf = ItemCFBaseline(train_seq)
    all_metrics['ItemCF'] = evaluate_baseline(
        'ItemCF', itemcf.recommend, test_seq, ks)

    # 3. SASRec baseline
    logger.info('Running SASRec baseline...')
    # Build item index mapping
    all_items_set = set(all_item_ids)
    item_to_idx = {item: i + 1 for i, item in enumerate(sorted(all_items_set))}
    idx_to_item = {i + 1: item for i, item in enumerate(sorted(all_items_set))}
    idx_to_item[0] = ''  # padding idx

    if len(train_seq) > 0:
        sasrec_model = SASRecBaseline(
            num_items=len(item_to_idx) + 1,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=args.max_history_len,
        )
        sasrec_model = train_sasrec(
            sasrec_model, train_seq, item_to_idx,
            epochs=20, batch_size=128, lr=1e-3, device=device,
        )

        def sasrec_recommend(history, k):
            return sasrec_model.recommend(history, item_to_idx, idx_to_item, k)

        all_metrics['SASRec'] = evaluate_baseline(
            'SASRec', sasrec_recommend, test_seq, ks)
    else:
        logger.warning('No training sequences for SASRec; skipping.')

    # 4. Random SID generator baseline
    logger.info('Running Random SID generator baseline...')
    random_sid = RandomSIDGeneratorBaseline(
        sid_to_items, args.num_sid_tokens, args.vocab_per_token)

    all_metrics['RandomSID'] = evaluate_baseline(
        'RandomSID', random_sid.recommend, test_seq, ks)

    # 5. Trained SID generator baseline (if checkpoint provided)
    if args.sid_generator_ckpt and os.path.exists(args.sid_generator_ckpt):
        logger.info(f'Running Trained SID Generator baseline...')
        trie = build_sid_trie(sid_to_items)
        trained_sid = TrainedSIDGeneratorBaseline(
            args.sid_generator_ckpt, trie, args.beam_width, device)

        def trained_recommend(history_items, k):
            history_sids = [item_to_sid.get(i) for i in history_items if i in item_to_sid]
            history_sids = [s for s in history_sids if s is not None]
            if not history_sids:
                # Fall back to popularity
                return pop_baseline.recommend(history_items, k)
            return trained_sid.recommend(history_sids, sid_to_items, k)

        all_metrics['SIDGen+Beam'] = evaluate_baseline(
            'SIDGen+Beam', trained_recommend, test_seq, ks)
    else:
        logger.info('No SID generator checkpoint; skipping SIDGen+Beam baseline.')

    # Print comparison table
    print_comparison_table(all_metrics, ks)

    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
