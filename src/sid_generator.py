"""
SID generator: small Transformer decoder for generating SID token sequences.

Supports:
    - Small Transformer decoder (Task 2.1)
    - Seen-item filtering (Task 2.4)
    - Duplicate filtering (Task 2.4)
    - Latte-style latent tokens (Task 4.1, 4.2)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SIDGeneratorConfig:
    """Configuration for the SID generator model.

    Attributes:
        vocab_size_per_token: number of unique tokens per SID position.
        num_sid_tokens: number of tokens in each SID sequence.
        max_history_len: maximum number of history items to consider.
        hidden_dim: hidden dimension of the Transformer.
        num_layers: number of Transformer decoder layers.
        num_heads: number of attention heads.
        dropout: dropout probability.
        max_seq_len: maximum total sequence length (history + SID tokens).
        use_latent_tokens: enable Latte-style latent tokens (Task 4.1).
        latent_token_count: number of latent tokens (default 0 = disabled).
        latent_token_dim: dimension of latent tokens (default = hidden_dim).
    """
    vocab_size_per_token: int = 256
    num_sid_tokens: int = 3
    max_history_len: int = 50
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 512
    use_latent_tokens: bool = False
    latent_token_count: int = 0
    latent_token_dim: Optional[int] = None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SIDGenerator(nn.Module):
    """Small Transformer decoder for SID token generation.

    Architecture:
        - Token embedding layer (shared across all SID positions)
        - Positional encoding
        - N-layer Transformer decoder (causal masking)
        - Output projection head per SID token position

    With latent tokens (Latte-style, Task 4.1):
        - Learnable latent tokens are prepended before SID generation
        - Latent tokens attend to history (cross-attention not needed if
          we simply concatenate them in the sequence with causal masking)
        - SID tokens attend to latent tokens + history
    """

    def __init__(self, config=None, vocab_per_token=256, num_sid_tokens=3,
                 hidden_dim=128, num_layers=3, num_heads=4, max_len=50,
                 use_latent_tokens=False, latent_token_count=0):
        """Initialize SIDGenerator.

        Supports two calling conventions:
            1) SIDGenerator(config=SIDGeneratorConfig(...))   -- dataclass config
            2) SIDGenerator(num_sid_tokens=3, vocab_per_token=256, ...)  -- individual params
            3) SIDGenerator(3, 256)  -- positional: (num_sid_tokens, vocab_per_token)
        """
        super().__init__()
        if isinstance(config, SIDGeneratorConfig):
            self.config = config
        elif isinstance(config, int) or config is None:
            # Positional or keyword-based construction
            actual_num_sid = num_sid_tokens if config is None else config
            actual_vocab = vocab_per_token
            self.config = SIDGeneratorConfig(
                vocab_size_per_token=actual_vocab,
                num_sid_tokens=actual_num_sid,
                max_history_len=max_len,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                use_latent_tokens=use_latent_tokens,
                latent_token_count=latent_token_count if use_latent_tokens else 0,
            )
        else:
            raise TypeError(
                f"Expected SIDGeneratorConfig, int, or None for config, got {type(config)}"
            )
        self.hidden_dim = self.config.hidden_dim

        # Token embedding (shared across all SID positions)
        self.token_embedding = nn.Embedding(
            self.config.vocab_size_per_token, self.config.hidden_dim
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.config.hidden_dim, self.config.max_seq_len)

        # Learnable special tokens
        self.history_sep_token = nn.Parameter(
            torch.randn(1, 1, self.config.hidden_dim) * 0.02
        )
        self.eos_token_embed = nn.Parameter(
            torch.randn(1, 1, self.config.hidden_dim) * 0.02
        )

        # Latent tokens (Task 4.1)
        self.use_latent_tokens = self.config.use_latent_tokens
        if self.use_latent_tokens and self.config.latent_token_count > 0:
            latent_dim = self.config.latent_token_dim or self.config.hidden_dim
            self.latent_tokens = nn.Parameter(
                torch.randn(1, self.config.latent_token_count, latent_dim) * 0.02
            )
            # Project if dimensions don't match
            if latent_dim != self.config.hidden_dim:
                self.latent_proj = nn.Linear(latent_dim, self.config.hidden_dim)
            else:
                self.latent_proj = nn.Identity()
        else:
            self.latent_tokens = None
            self.latent_proj = None

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="relu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_layers)

        # Output heads: one linear per SID token position
        self.output_heads = nn.ModuleList([
            nn.Linear(self.config.hidden_dim, self.config.vocab_size_per_token)
            for _ in range(self.config.num_sid_tokens)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        history_sids: torch.Tensor,
        target_sids: Optional[torch.Tensor] = None,
        num_latent: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            history_sids: (batch, history_len, num_sid_tokens) history item SIDs.
            target_sids: optional (batch, target_len, num_sid_tokens) teacher-forcing targets.
            num_latent: override latent_token_count for this forward pass.

        Returns:
            Dict with keys:
                "logits": (batch, target_len, num_sid_tokens, vocab_size_per_token)
                          per-position logits if target_sids is provided.
                "latent_z": latent representations if latent tokens are used.
        """
        B, H, T = history_sids.shape
        device = history_sids.device

        # Flatten history SIDs: (B, H*T) token indices
        history_flat = history_sids.reshape(B, H * T)

        # Embed history tokens
        history_emb = self.token_embedding(history_flat)  # (B, H*T, D)

        # Build history segment with separator
        sep = self.history_sep_token.expand(B, 1, -1)
        memory = torch.cat([history_emb, sep], dim=1)  # (B, H*T + 1, D)

        # Build target / query sequence
        if target_sids is not None:
            B2, L, T2 = target_sids.shape
            target_flat = target_sids.reshape(B2, L * T2)
            target_emb = self.token_embedding(target_flat)  # (B, L*T, D)
            # Shift right for teacher forcing: prepend start token (zeros)
            start = torch.zeros(B2, 1, self.hidden_dim, device=device)
            tgt = torch.cat([start, target_emb[:, :-1, :]], dim=1)
        else:
            # Autoregressive generation: start with zeros
            tgt = torch.zeros(B, 1, self.hidden_dim, device=device)

        # Handle latent tokens (Task 4.1/4.2)
        latent_count = num_latent if num_latent is not None else self.config.latent_token_count
        latent_emb = None
        if self.use_latent_tokens and latent_count > 0 and self.latent_tokens is not None:
            # Expand latent tokens to batch
            latent_emb = self.latent_tokens.expand(B, -1, -1)
            latent_emb = self.latent_proj(latent_emb)

            # Concatenate before the target sequence
            tgt = torch.cat([latent_emb, tgt], dim=1)

        # Apply positional encoding
        tgt = self.pos_encoder(tgt)
        memory = self.pos_encoder(memory)

        # Create causal mask for decoder self-attention
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float("-inf"), device=device),
            diagonal=1,
        )

        # Decoder forward
        output = self.decoder(
            tgt, memory,
            tgt_mask=causal_mask,
        )  # (B, tgt_len, D)

        result = {}

        # Extract latent representations if used
        if latent_emb is not None:
            result["latent_z"] = output[:, :latent_count, :]

        # Compute per-position logits
        logits_list = []
        start_idx = latent_count if latent_emb is not None else 0

        for i in range(self.config.num_sid_tokens):
            pos_logits = self.output_heads[i](output[:, start_idx + i:start_idx + i + 1, :])
            logits_list.append(pos_logits)  # (B, 1, vocab)

        logits = torch.cat(logits_list, dim=1)  # (B, num_sid_tokens, vocab)

        if target_sids is not None:
            # Reshape to (B, L, num_sid_tokens, vocab) for multiple target positions
            # Currently logits is (B, num_sid_tokens, vocab) for single-step prediction
            # For multi-step teacher forcing, we need to handle it differently
            result["logits"] = logits.unsqueeze(1)  # (B, 1, num_sid_tokens, vocab)
        else:
            result["logits"] = logits.unsqueeze(1)

        return result

    @torch.no_grad()
    def generate_single_step(
        self,
        history_sids: torch.Tensor,
        step_input: Optional[torch.Tensor] = None,
        num_latent: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate the next SID probabilities given history and optional partial SID.

        Args:
            history_sids: (batch, history_len, num_sid_tokens) history.
            step_input: optional (batch, 1, num_sid_tokens) previously generated SID.
            num_latent: override latent token count.

        Returns:
            torch.Tensor: (batch, num_sid_tokens, vocab_size) logits.
        """
        self.eval()
        out = self.forward(
            history_sids=history_sids,
            target_sids=step_input,
            num_latent=num_latent,
        )
        return out["logits"].squeeze(1)  # (B, num_sid_tokens, vocab)

    @torch.no_grad()
    def get_next_token_logits(
        self,
        history_sids: torch.Tensor,
        current_sid_prefix: Optional[List[int]] = None,
        num_latent: Optional[int] = None,
    ) -> torch.Tensor:
        """Get logits for the next SID token position.

        Args:
            history_sids: (1, history_len, num_sid_tokens) single user history.
            current_sid_prefix: partial SID tokens generated so far.

        Returns:
            torch.Tensor: (vocab_size,) logits for the next token.
        """
        B = history_sids.shape[0]

        if current_sid_prefix is None or len(current_sid_prefix) == 0:
            step_input = None
        else:
            prefix_tensor = torch.tensor(
                [[current_sid_prefix]], dtype=torch.long, device=history_sids.device
            )  # (1, 1, len(prefix))
            if prefix_tensor.size(-1) < self.config.num_sid_tokens:
                pad_len = self.config.num_sid_tokens - prefix_tensor.size(-1)
                pad = torch.zeros(1, 1, pad_len, dtype=torch.long, device=history_sids.device)
                prefix_tensor = torch.cat([prefix_tensor, pad], dim=-1)
            step_input = prefix_tensor

        logits = self.generate_single_step(history_sids, step_input, num_latent)
        pos = len(current_sid_prefix) if current_sid_prefix else 0
        pos = min(pos, self.config.num_sid_tokens - 1)
        return logits[0, pos, :]  # (vocab_size,)


# ===== Filtering utilities (Task 2.4) =====


def filter_seen_items(
    generated_sids: List[Tuple[int, ...]],
    user_history_sids: List[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    """Remove SIDs that the user has already interacted with.

    Args:
        generated_sids: list of generated SID tuples.
        user_history_sids: list of SID tuples from user history.

    Returns:
        Filtered list with seen SIDs removed (preserving order).
    """
    seen = set(user_history_sids)
    return [sid for sid in generated_sids if sid not in seen]


def filter_duplicates(
    candidates: List[Tuple[Any, ...]],
) -> List[Tuple[Any, ...]]:
    """Deduplicate candidate list while preserving order.

    Args:
        candidates: list of (item_id, score) or just item_ids.

    Returns:
        Deduplicated list preserving order of first occurrence.
    """
    seen: Set[Any] = set()
    result: List[Tuple[Any, ...]] = []
    for c in candidates:
        key = c[0] if isinstance(c, (list, tuple)) else c
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result
