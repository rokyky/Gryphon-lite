"""
Item-level scorers for Gryphon-style candidate reranking (Tasks 3.2, 3.3).

DotProductScorer: simple dot-product between user and item embeddings.
MLPScorer: MLP over user, item, and SID embeddings with optional features.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ItemScorerConfig:
    """Configuration for item scorers.

    Attributes:
        scorer_type: "dotproduct" or "mlp".
        user_dim: dimension of user embeddings.
        item_dim: dimension of item embeddings.
        sid_dim: dimension of SID embeddings (sum of token embeddings).
        hidden_dims: hidden layer dimensions for MLP scorer.
        dropout: dropout for MLP scorer.
        activation: activation function (relu, gelu, tanh).
        use_category_feature: whether to include category match feature.
        use_popularity_feature: whether to include popularity feature.
        use_recency_feature: whether to include recency feature.
    """
    scorer_type: str = "mlp"
    user_dim: int = 128
    item_dim: int = 128
    sid_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    activation: str = "relu"
    use_category_feature: bool = False
    use_popularity_feature: bool = False
    use_recency_feature: bool = False


class DotProductScorer(nn.Module):
    """Simple dot-product scorer (Task 3.2).

    score = dot(user_emb, item_emb)

    User embedding: mean pooling over history item embeddings.
    Item embedding: from item metadata or learned lookup.
    """

    def __init__(
        self,
        user_dim: int = 128,
        item_dim: int = 128,
        num_items: int = 0,
        use_learned_item_emb: bool = True,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.use_learned_item_emb = use_learned_item_emb

        if use_learned_item_emb and num_items > 0:
            self.item_embedding = nn.Embedding(num_items, item_dim)
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        else:
            self.item_embedding = None

        # Projection if dimensions differ
        if user_dim != item_dim:
            self.user_proj = nn.Linear(user_dim, item_dim)
        else:
            self.user_proj = nn.Identity()

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute dot-product scores.

        Args:
            user_embeddings: (batch, user_dim) or (batch, history_len, user_dim).
            item_embeddings: (batch, num_candidates, item_dim) or None.
            item_ids: optional (batch, num_candidates) for learned embeddings.

        Returns:
            torch.Tensor: (batch, num_candidates) scores.
        """
        # Mean pool user history if needed
        if user_embeddings.dim() == 3:
            user_emb = user_embeddings.mean(dim=1)  # (batch, user_dim)
        else:
            user_emb = user_embeddings

        user_emb = self.user_proj(user_emb)  # (batch, item_dim)

        # Get item embeddings
        if item_embeddings is not None:
            item_emb = item_embeddings
        elif self.item_embedding is not None and item_ids is not None:
            item_emb = self.item_embedding(item_ids)  # (batch, num_candidates, item_dim)
        else:
            raise ValueError("Either item_embeddings or learned item_embedding is required.")

        # Dot product
        scores = torch.bmm(
            item_emb, user_emb.unsqueeze(-1)
        ).squeeze(-1)  # (batch, num_candidates)

        return scores

    @torch.no_grad()
    def compute_user_embedding(
        self,
        history_item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute user embedding as mean of history item embeddings."""
        if history_item_embeddings.dim() == 3:
            return history_item_embeddings.mean(dim=1)
        return history_item_embeddings


class MLPScorer(nn.Module):
    """MLP-based scorer (Task 3.3).

    score = MLP([user_emb, item_emb, sid_emb, optional_features])

    Architecture:
        - Concatenate user, item, and SID embeddings
        - Optionally add category match, popularity, recency features
        - 2-3 layer MLP with ReLU/GELU
        - Single output neuron (score)
    """

    def __init__(self, config: ItemScorerConfig, num_items: int = 0):
        super().__init__()
        self.config = config

        # Compute input dimension
        feature_dims = config.user_dim + config.item_dim + config.sid_dim
        extra_features = 0
        if config.use_category_feature:
            extra_features += 1
        if config.use_popularity_feature:
            extra_features += 1
        if config.use_recency_feature:
            extra_features += 1
        self.extra_features = extra_features
        input_dim = feature_dims + extra_features

        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Dropout(config.dropout))
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Optional learned embeddings
        if num_items > 0:
            self.item_embedding = nn.Embedding(num_items, config.item_dim)
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        else:
            self.item_embedding = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: Optional[torch.Tensor] = None,
        sid_embeddings: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
        extra_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute MLP scores.

        Args:
            user_embeddings: (batch, user_dim) or (batch, history_len, user_dim).
            item_embeddings: (batch, num_candidates, item_dim) or None.
            sid_embeddings: (batch, num_candidates, sid_dim) or None.
            item_ids: optional (batch, num_candidates) for learned embeddings.
            extra_features: optional (batch, num_candidates, extra_feat_dim).

        Returns:
            torch.Tensor: (batch, num_candidates) scores.
        """
        B = user_embeddings.shape[0]

        # Mean pool user history if needed
        if user_embeddings.dim() == 3:
            user_emb = user_embeddings.mean(dim=1)  # (B, user_dim)
        else:
            user_emb = user_embeddings

        # Determine num_candidates
        if item_embeddings is not None:
            num_candidates = item_embeddings.size(1)
        elif item_ids is not None:
            num_candidates = item_ids.size(1)
        else:
            num_candidates = 1

        # Expand user embedding to match candidates
        user_emb_expanded = user_emb.unsqueeze(1).expand(B, num_candidates, -1)

        # Get item embeddings
        if item_embeddings is None and self.item_embedding is not None and item_ids is not None:
            item_embeddings = self.item_embedding(item_ids)
        elif item_embeddings is None:
            item_embeddings = torch.zeros(B, num_candidates, self.config.item_dim, device=user_emb.device)

        # Default sid embeddings if missing
        if sid_embeddings is None:
            sid_embeddings = torch.zeros(B, num_candidates, self.config.sid_dim, device=user_emb.device)

        # Concatenate features
        concat_list = [user_emb_expanded, item_embeddings, sid_embeddings]

        if extra_features is not None:
            concat_list.append(extra_features)

        combined = torch.cat(concat_list, dim=-1)  # (B, num_candidates, input_dim)

        # MLP forward
        scores = self.mlp(combined).squeeze(-1)  # (B, num_candidates)

        return scores


def create_item_scorer(
    config: ItemScorerConfig,
    num_items: int = 0,
) -> nn.Module:
    """Factory: create an item scorer by type."""
    if config.scorer_type == "dotproduct":
        return DotProductScorer(
            user_dim=config.user_dim,
            item_dim=config.item_dim,
            num_items=num_items,
        )
    elif config.scorer_type == "mlp":
        return MLPScorer(config, num_items=num_items)
    else:
        raise ValueError(f"Unknown scorer type: {config.scorer_type}")
