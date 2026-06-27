#!/usr/bin/env python3
"""
Train the Gryphon-style item-level scorer (Task 3.4).

Training data:
    - Positives: next-item from user history
    - Hard negatives: generated SID candidates that are NOT the target
    - Random negatives: random items from catalog

Loss: binary cross-entropy or pairwise ranking loss.

Usage:
    python scripts/train_item_scorer.py \\
        --train_path data/train.csv \\
        --index_path data/indices.json \\
        --output_dir checkpoints/item_scorer \\
        --epochs 30 --batch_size 32 --lr 1e-3
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.item_scorer import ItemScorerConfig, create_item_scorer
from src.sid_generator import SIDGenerator, SIDGeneratorConfig
from src.sid_mapper import SIDTrie, build_sid_trie
from src.trie_constrained_decoder import TrieConstrainedBeamSearch, TrieBeamSearchConfig

logger = logging.getLogger(__name__)


class ItemScorerTrainDataset(Dataset):
    """Dataset for item scorer training.

    Each sample contains:
        - user_history_sids: SID sequence for user history
        - positive_item_id: the next item the user interacted with
        - hard_negative_item_ids: generated candidates that are NOT the target
        - random_negative_item_ids: random items from catalog
    """

    def __init__(
        self,
        data_path: str,
        index_path: str,
        num_random_negatives: int = 100,
        max_history_len: int = 50,
        num_sid_tokens: int = 3,
        min_seq_len: int = 2,
    ):
        self.num_random_negatives = num_random_negatives
        self.max_history_len = max_history_len
        self.num_sid_tokens = num_sid_tokens
        self.min_seq_len = min_seq_len

        # Load index
        with open(index_path, "r") as f:
            self.index: Dict[str, List[int]] = json.load(f)

        self.item_to_sid: Dict[str, Tuple[int, ...]] = {}
        for item_id, sid_list in self.index.items():
            self.item_to_sid[item_id] = tuple(int(s) for s in sid_list)

        self.all_item_ids = list(self.item_to_sid.keys())

        # Load sequences
        import pandas as pd
        self.data = pd.read_csv(data_path)

        # Build samples
        self.samples: List[Dict[str, Any]] = []
        self._build_samples()

    def _build_samples(self):
        for idx in tqdm(range(len(self.data)), desc="Building scorer samples"):
            row = self.data.iloc[idx]

            try:
                history_item_ids = eval(str(row.get("history_item_id", "[]")))
            except (ValueError, SyntaxError):
                continue

            target_item_id = str(row.get("item_id", ""))
            if len(history_item_ids) < self.min_seq_len or not target_item_id:
                continue

            if target_item_id not in self.item_to_sid:
                continue

            history_sids = []
            for h_id in history_item_ids:
                sid = self.item_to_sid.get(str(h_id))
                if sid is not None:
                    history_sids.append(sid)

            if len(history_sids) < self.min_seq_len:
                continue

            # Truncate history
            if len(history_sids) > self.max_history_len:
                history_sids = history_sids[-self.max_history_len:]

            # Random negatives (sample items not in history)
            history_set = set(str(h) for h in history_item_ids)
            valid_negatives = [
                iid for iid in self.all_item_ids
                if iid != target_item_id and iid not in history_set
            ]

            n_neg = min(self.num_random_negatives, len(valid_negatives))
            random_negatives = random.sample(valid_negatives, n_neg) if n_neg > 0 else []

            self.samples.append({
                "history_sids": history_sids,
                "history_item_ids": [str(h) for h in history_item_ids],
                "positive_item_id": target_item_id,
                "random_negative_ids": random_negatives,
            })

        logger.info(f"Built {len(self.samples)} scorer training samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_scorer_samples(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch of scorer samples.

    Returns collated data with padded negative lists.
    """
    max_negatives = max(len(s["random_negative_ids"]) for s in batch)

    collated = {
        "history_sids": [],
        "pos_item_ids": [],
        "neg_item_ids": [],
    }

    for s in batch:
        collated["history_sids"].append(s["history_sids"])
        collated["pos_item_ids"].append(s["positive_item_id"])

        negs = s["random_negative_ids"]
        # Pad negatives
        while len(negs) < max_negatives:
            negs.append(negs[0] if negs else s["positive_item_id"])
        collated["neg_item_ids"].append(negs)

    return collated


def extract_sid_embedding(
    sid: Tuple[int, ...],
    model: SIDGenerator,
    device: torch.device,
) -> torch.Tensor:
    """Extract SID embedding from generator token embeddings."""
    sid_tensor = torch.tensor([sid], dtype=torch.long, device=device)
    emb = model.token_embedding(sid_tensor)  # (1, T, D)
    return emb.mean(dim=1)  # (1, D)


def train_epoch(
    scorer: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sid_generator: Optional[SIDGenerator] = None,
    item_to_sid: Optional[Dict[str, Tuple[int, ...]]] = None,
    user_dim: int = 128,
) -> float:
    """Train for one epoch using pairwise ranking loss."""
    scorer.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training scorer"):
        optimizer.zero_grad()

        batch_loss = 0.0
        num_samples = len(batch["pos_item_ids"])

        for i in range(num_samples):
            history_sids = batch["history_sids"][i]
            pos_id = batch["pos_item_ids"][i]
            neg_ids = batch["neg_item_ids"][i]

            # Pad history
            pad_len = 50 - len(history_sids)
            if pad_len > 0:
                pad_sid = tuple([0] * 3)
                history_sids_padded = [pad_sid] * pad_len + history_sids
            else:
                history_sids_padded = history_sids[-50:]

            hist_tensor = torch.tensor([history_sids_padded], dtype=torch.long, device=device)

            # Get user embedding from SID generator
            with torch.no_grad():
                if sid_generator is not None:
                    user_emb = sid_generator.token_embedding(
                        hist_tensor.reshape(1, -1)
                    ).mean(dim=1)  # (1, D)
                else:
                    user_emb = torch.zeros(1, user_dim, device=device)

            # Positive score
            pos_sid = item_to_sid.get(pos_id, (0,) * 3)
            pos_sid_emb = extract_sid_embedding(pos_sid, sid_generator, device) if sid_generator else None

            pos_item_id_tensor = torch.tensor([[hash(pos_id) % 10000]], device=device)
            pos_score = scorer(
                user_embeddings=user_emb,
                item_ids=pos_item_id_tensor,
                sid_embeddings=pos_sid_emb.unsqueeze(1) if pos_sid_emb is not None else None,
            )

            # Negative scores (mean over negatives)
            neg_scores_list = []
            for neg_id in neg_ids:
                neg_sid = item_to_sid.get(neg_id, (0,) * 3)
                neg_sid_emb = extract_sid_embedding(neg_sid, sid_generator, device) if sid_generator else None

                neg_item_id_tensor = torch.tensor([[hash(neg_id) % 10000]], device=device)
                neg_score = scorer(
                    user_embeddings=user_emb,
                    item_ids=neg_item_id_tensor,
                    sid_embeddings=neg_sid_emb.unsqueeze(1) if neg_sid_emb is not None else None,
                )
                neg_scores_list.append(neg_score)

            neg_scores = torch.stack(neg_scores_list)

            # BPR pairwise loss: -log(sigmoid(pos - neg))
            # Average over negatives
            pairwise_loss = -F.logsigmoid(pos_score - neg_scores).mean()

            batch_loss += pairwise_loss

        batch_loss = batch_loss / num_samples
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(scorer, dataloader, device, sid_generator=None, item_to_sid=None):
    """Validation: compute mean score margin (pos - neg)."""
    scorer.eval()
    margins = []
    num_samples = 0

    for batch in dataloader:
        for i in range(len(batch["pos_item_ids"])):
            history_sids = batch["history_sids"][i]
            pos_id = batch["pos_item_ids"][i]
            neg_ids = batch["neg_item_ids"][i]

            pad_len = 50 - len(history_sids)
            if pad_len > 0:
                pad_sid = tuple([0] * 3)
                history_sids_padded = [pad_sid] * pad_len + history_sids
            else:
                history_sids_padded = history_sids[-50:]

            hist_tensor = torch.tensor([history_sids_padded], dtype=torch.long, device=device)

            user_emb = sid_generator.token_embedding(
                hist_tensor.reshape(1, -1)
            ).mean(dim=1) if sid_generator else torch.zeros(1, 128, device=device)

            pos_sid = item_to_sid.get(pos_id, (0,) * 3)
            pos_sid_emb = extract_sid_embedding(pos_sid, sid_generator, device) if sid_generator else None
            pos_id_tensor = torch.tensor([[hash(pos_id) % 10000]], device=device)
            pos_score = scorer(
                user_embeddings=user_emb,
                item_ids=pos_id_tensor,
                sid_embeddings=pos_sid_emb.unsqueeze(1) if pos_sid_emb is not None else None,
            )

            neg_scores = []
            for neg_id in neg_ids[:10]:  # subsample for speed
                neg_sid = item_to_sid.get(neg_id, (0,) * 3)
                neg_sid_emb = extract_sid_embedding(neg_sid, sid_generator, device) if sid_generator else None
                neg_id_tensor = torch.tensor([[hash(neg_id) % 10000]], device=device)
                ns = scorer(
                    user_embeddings=user_emb,
                    item_ids=neg_id_tensor,
                    sid_embeddings=neg_sid_emb.unsqueeze(1) if neg_sid_emb is not None else None,
                )
                neg_scores.append(ns.item())

            if neg_scores:
                margins.append(pos_score.item() - np.mean(neg_scores))
            num_samples += 1

    avg_margin = np.mean(margins) if margins else 0.0
    return avg_margin


def parse_args():
    parser = argparse.ArgumentParser(description="Train item-level scorer")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--sid_generator_ckpt", type=str, default=None,
                        help="Pretrained SID generator checkpoint")
    parser.add_argument("--sid_generator_config", type=str, default=None)

    parser.add_argument("--scorer_type", type=str, default="mlp", choices=["dotproduct", "mlp"])
    parser.add_argument("--user_dim", type=int, default=128)
    parser.add_argument("--item_dim", type=int, default=128)
    parser.add_argument("--sid_dim", type=int, default=128)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_random_negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, default="checkpoints/item_scorer")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load index
    with open(args.index_path, "r") as f:
        index = json.load(f)
    item_to_sid = {iid: tuple(int(s) for s in sid_list) for iid, sid_list in index.items()}
    num_items = len(item_to_sid)

    # Load SID generator (optional, for embeddings)
    sid_generator = None
    if args.sid_generator_ckpt:
        logger.info(f"Loading SID generator from {args.sid_generator_ckpt}")
        ckpt = torch.load(args.sid_generator_ckpt, map_location="cpu")
        cfg = ckpt.get("config", SIDGeneratorConfig())
        sid_generator = SIDGenerator(cfg).to(device)
        sid_generator.load_state_dict(ckpt["model_state_dict"])
        sid_generator.eval()

    # Build datasets
    train_dataset = ItemScorerTrainDataset(
        args.train_path, args.index_path,
        num_random_negatives=args.num_random_negatives,
    )
    if args.valid_path:
        valid_dataset = ItemScorerTrainDataset(
            args.valid_path, args.index_path,
            num_random_negatives=50,
        )
    else:
        n = len(train_dataset)
        n_valid = max(1, int(n * 0.1))
        n_train = n - n_valid
        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [n_train, n_valid],
        )
        logger.info(f"Split: {n_train} train, {n_valid} valid")

    def collate(batch):
        return collate_scorer_samples(batch)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate)

    # Create scorer
    scorer_config = ItemScorerConfig(
        scorer_type=args.scorer_type,
        user_dim=args.user_dim,
        item_dim=args.item_dim,
        sid_dim=args.sid_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    scorer = create_item_scorer(scorer_config, num_items=min(num_items, 100000)).to(device)

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_valid_margin = float("-inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            scorer, train_loader, optimizer, device,
            sid_generator=sid_generator, item_to_sid=item_to_sid,
        )
        valid_margin = validate(
            scorer, valid_loader, device,
            sid_generator=sid_generator, item_to_sid=item_to_sid,
        )
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
            f"Valid Margin: {valid_margin:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if valid_margin > best_valid_margin:
            best_valid_margin = valid_margin
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": scorer.state_dict(),
                "config": scorer_config,
                "valid_margin": valid_margin,
            }, ckpt_path)
            logger.info(f"Saved best model: {ckpt_path}")

    logger.info(f"Training complete. Best valid margin: {best_valid_margin:.4f}")


if __name__ == "__main__":
    main()
