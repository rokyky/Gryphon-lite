#!/usr/bin/env python3
"""
Train the SID generator with teacher-forced next-token prediction (Task 2.2).

Data: user sequences converted to SID token sequences.
Loss: cross-entropy per SID token position.

Usage:
    python scripts/train_sid_generator.py \\
        --data_path data/train.csv \\
        --index_path data/indices.json \\
        --output_dir checkpoints/sid_generator \\
        --epochs 50 --batch_size 64 --lr 1e-3
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sid_generator import SIDGenerator, SIDGeneratorConfig

logger = logging.getLogger(__name__)


class SidSequenceDataset(Dataset):
    """Dataset for teacher-forced SID generator training.

    Each sample: (history_sids, target_sid) from user interaction sequences.
    """

    def __init__(
        self,
        data_path: str,
        index_path: str,
        max_history_len: int = 50,
        num_sid_tokens: int = 3,
        min_seq_len: int = 2,
    ):
        self.max_history_len = max_history_len
        self.num_sid_tokens = num_sid_tokens
        self.min_seq_len = min_seq_len

        # Load index (item_id -> SID)
        with open(index_path, "r") as f:
            self.index: Dict[str, List[int]] = json.load(f)

        # Convert SID strings to int tuples
        self.item_to_sid: Dict[str, Tuple[int, ...]] = {}
        for item_id, sid_list in self.index.items():
            sid = tuple(int(s) for s in sid_list)
            self.item_to_sid[item_id] = sid

        # Load interaction sequences
        import pandas as pd
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} sequences from {data_path}")

        # Build samples
        self.samples: List[Dict[str, Any]] = []
        self._build_samples()

    def _build_samples(self):
        """Convert CSV rows to (history_sids, target_sid) pairs."""
        for idx in tqdm(range(len(self.data)), desc="Building SID sequences"):
            row = self.data.iloc[idx]

            # Parse history item IDs
            try:
                history_item_ids = eval(str(row.get("history_item_id", "[]")))
            except (ValueError, SyntaxError):
                history_item_ids = []

            # Get target item ID
            target_item_id = str(row.get("item_id", ""))

            if len(history_item_ids) < self.min_seq_len or not target_item_id:
                continue

            # Convert to SIDs
            history_sids = []
            for h_id in history_item_ids:
                sid = self.item_to_sid.get(str(h_id))
                if sid is not None and len(sid) == self.num_sid_tokens:
                    history_sids.append(sid)

            target_sid = self.item_to_sid.get(target_item_id)
            if target_sid is None or len(target_sid) != self.num_sid_tokens:
                continue

            if len(history_sids) < self.min_seq_len:
                continue

            # Truncate history
            if len(history_sids) > self.max_history_len:
                history_sids = history_sids[-self.max_history_len:]

            self.samples.append({
                "history_sids": history_sids,
                "target_sid": target_sid,
            })

        logger.info(f"Built {len(self.samples)} training samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        history = sample["history_sids"]
        target = sample["target_sid"]

        # Pad history to max_history_len
        pad_len = self.max_history_len - len(history)
        if pad_len > 0:
            pad_sid = tuple([0] * self.num_sid_tokens)
            history = [pad_sid] * pad_len + history
        else:
            history = history[-self.max_history_len:]

        hist_tensor = torch.tensor(history, dtype=torch.long)  # (H, T)
        target_tensor = torch.tensor(target, dtype=torch.long)  # (T,)

        return {
            "history_sids": hist_tensor,
            "target_sid": target_tensor,
        }


def collate_sid_sequences(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for SID sequence data."""
    history_sids = torch.stack([b["history_sids"] for b in batch])  # (B, H, T)
    target_sid = torch.stack([b["target_sid"] for b in batch])  # (B, T)
    return {
        "history_sids": history_sids,
        "target_sid": target_sid,
    }


def train_epoch(
    model: SIDGenerator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training"):
        history = batch["history_sids"].to(device)
        target = batch["target_sid"].to(device)

        optimizer.zero_grad()

        # Forward: teacher forcing
        # target needs to be (B, 1, T) for the model
        target_input = target.unsqueeze(1)  # (B, 1, T)
        output = model(history_sids=history, target_sids=target_input)

        # Compute cross-entropy loss per SID token
        logits = output["logits"]  # (B, 1, T, V)
        logits = logits.squeeze(1)  # (B, T, V)

        loss = 0.0
        for t in range(model.config.num_sid_tokens):
            loss += F.cross_entropy(logits[:, t, :], target[:, t])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: SIDGenerator,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validation"):
        history = batch["history_sids"].to(device)
        target = batch["target_sid"].to(device)

        target_input = target.unsqueeze(1)
        output = model(history_sids=history, target_sids=target_input)
        logits = output["logits"].squeeze(1)

        loss = 0.0
        for t in range(model.config.num_sid_tokens):
            loss += F.cross_entropy(logits[:, t, :], target[:, t])

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SID generator with teacher-forced next-token prediction"
    )
    # Data
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to training CSV")
    parser.add_argument("--valid_path", type=str, default=None,
                        help="Path to validation CSV")
    parser.add_argument("--index_path", type=str, required=True,
                        help="Path to index.json (item_id -> SID)")

    # Model
    parser.add_argument("--num_sid_tokens", type=int, default=3)
    parser.add_argument("--vocab_size_per_token", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_history_len", type=int, default=50)
    parser.add_argument("--use_latent_tokens", action="store_true")
    parser.add_argument("--latent_token_count", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/sid_generator")
    parser.add_argument("--save_every", type=int, default=10)

    # Device
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    train_dataset = SidSequenceDataset(
        data_path=args.train_path,
        index_path=args.index_path,
        max_history_len=args.max_history_len,
        num_sid_tokens=args.num_sid_tokens,
    )

    if args.valid_path:
        valid_dataset = SidSequenceDataset(
            data_path=args.valid_path,
            index_path=args.index_path,
            max_history_len=args.max_history_len,
            num_sid_tokens=args.num_sid_tokens,
        )
    else:
        # Split train into train/valid
        n = len(train_dataset)
        n_valid = max(1, int(n * 0.1))
        n_train = n - n_valid
        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [n_train, n_valid],
            generator=torch.Generator().manual_seed(args.seed),
        )
        logger.info(f"Split {n} samples: {n_train} train, {n_valid} valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_sid_sequences,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_sid_sequences,
    )

    # Build model
    config = SIDGeneratorConfig(
        vocab_size_per_token=args.vocab_size_per_token,
        num_sid_tokens=args.num_sid_tokens,
        max_history_len=args.max_history_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_latent_tokens=args.use_latent_tokens,
        latent_token_count=args.latent_token_count if args.use_latent_tokens else 0,
    )
    model = SIDGenerator(config).to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params:,} params ({trainable_params:,} trainable)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training loop
    best_valid_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        valid_loss = validate(model, valid_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save checkpoint
        if epoch % args.save_every == 0 or valid_loss < best_valid_loss:
            is_best = valid_loss < best_valid_loss
            if is_best:
                best_valid_loss = valid_loss

            ckpt_path = os.path.join(
                args.output_dir,
                f"epoch_{epoch}_loss_{valid_loss:.4f}.pt" if not is_best
                else os.path.join(args.output_dir, "best_model.pt"),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()
