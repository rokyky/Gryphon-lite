"""
Semantic ID (SID) builders for item-to-token-sequence mapping.

Supports multiple construction strategies:
    - RandomSIDBuilder: assign random token sequences as SIDs
    - CategoryAwareSIDBuilder: prefix category ID + random suffix
    - KMeansSIDBuilder: cluster text embeddings, use cluster IDs as SID tokens
    - RQKMeansSIDBuilder: residual quantized KMeans for multi-level SIDs

All builders expose:
    item_to_sid: dict[str or int, tuple[int, ...]]
    sid_to_items: dict[tuple[int, ...], list[str or int]]
"""

import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class BaseSIDBuilder(ABC):
    """Abstract base class for all SID builders."""

    def __init__(
        self,
        num_sid_tokens: int = 3,
        vocab_size_per_token: int = 256,
        seed: Optional[int] = None,
    ):
        self.num_sid_tokens = num_sid_tokens
        self.vocab_size_per_token = vocab_size_per_token
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.item_to_sid: Dict[Any, Tuple[int, ...]] = {}
        self.sid_to_items: Dict[Tuple[int, ...], List[Any]] = defaultdict(list)

    @abstractmethod
    def build(self, item_ids: List[Any], **kwargs) -> Tuple[Dict[Any, Tuple[int, ...]], Dict[Tuple[int, ...], List[Any]]]:
        """Build SID mappings for the given item ids."""
        ...

    def _finalize(self, item_ids: List[Any], sid_assignments: List[Tuple[int, ...]]):
        """Populate item_to_sid and sid_to_items from parallel lists."""
        self.item_to_sid.clear()
        self.sid_to_items.clear()
        for item_id, sid in zip(item_ids, sid_assignments):
            sid_tuple = tuple(sid)
            self.item_to_sid[item_id] = sid_tuple
            self.sid_to_items[sid_tuple].append(item_id)

        unique_sids = len(self.sid_to_items)
        total_items = len(item_ids)
        collisions = total_items - unique_sids
        logger.info(
            f"[{self.__class__.__name__}] Built {unique_sids} unique SIDs "
            f"for {total_items} items, {collisions} collisions "
            f"(rate={collisions / max(total_items, 1):.4f})"
        )
        return self.item_to_sid, dict(self.sid_to_items)


class RandomSIDBuilder(BaseSIDBuilder):
    """Assign uniformly random token sequences as SIDs."""

    def build(
        self,
        item_ids: List[Any],
        **kwargs,
    ) -> Tuple[Dict[Any, Tuple[int, ...]], Dict[Tuple[int, ...], List[Any]]]:
        n = len(item_ids)
        rng = random.Random(kwargs.get("seed", None))

        sid_assignments = []
        for _ in range(n):
            sid = tuple(rng.randint(0, self.vocab_size_per_token - 1)
                        for _ in range(self.num_sid_tokens))
            sid_assignments.append(sid)

        return self._finalize(item_ids, sid_assignments)


class CategoryAwareSIDBuilder(BaseSIDBuilder):
    """Prefix category ID + random suffix as SID.

    Requires `categories` kwarg: a list of category labels parallel to item_ids.
    """

    def __init__(
        self,
        num_sid_tokens: int = 3,
        vocab_size_per_token: int = 256,
        seed: Optional[int] = None,
        category_vocab: Optional[Dict[str, int]] = None,
    ):
        super().__init__(num_sid_tokens, vocab_size_per_token, seed)
        self.category_vocab = category_vocab or {}

    def build(
        self,
        item_ids: List[Any],
        categories: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[Dict[Any, Tuple[int, ...]], Dict[Tuple[int, ...], List[Any]]]:
        if categories is None:
            raise ValueError("CategoryAwareSIDBuilder requires `categories` kwarg.")

        # Build category vocab if not provided
        if not self.category_vocab:
            unique_cats = sorted(set(c for c in categories if c is not None))
            self.category_vocab = {cat: i for i, cat in enumerate(unique_cats)}
            if len(self.category_vocab) >= self.vocab_size_per_token:
                logger.warning(
                    f"Number of categories ({len(self.category_vocab)}) exceeds "
                    f"vocab_size_per_token ({self.vocab_size_per_token}). "
                    f"Some categories will collide."
                )

        rng = random.Random(kwargs.get("seed", None))
        sid_assignments = []
        for cat in categories:
            cat_id = self.category_vocab.get(cat, 0) % self.vocab_size_per_token
            suffix = tuple(
                rng.randint(0, self.vocab_size_per_token - 1)
                for _ in range(self.num_sid_tokens - 1)
            )
            sid_assignments.append((cat_id,) + suffix)

        return self._finalize(item_ids, sid_assignments)


class KMeansSIDBuilder(BaseSIDBuilder):
    """Cluster text embeddings via KMeans and use cluster IDs as SID tokens.

    This produces flat clustering: each level of SID tokens comes from
    a separate KMeans run on the same embeddings, so tokens at different
    positions capture complementary cluster structure.
    """

    def __init__(
        self,
        num_sid_tokens: int = 3,
        vocab_size_per_token: int = 256,
        seed: Optional[int] = None,
        kmeans_iters: int = 100,
        n_init: int = 10,
    ):
        super().__init__(num_sid_tokens, vocab_size_per_token, seed)
        self.kmeans_iters = kmeans_iters
        self.n_init = n_init
        self.cluster_models: List[KMeans] = []

    def build(
        self,
        item_ids: List[Any],
        embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[Dict[Any, Tuple[int, ...]], Dict[Tuple[int, ...], List[Any]]]:
        if embeddings is None:
            raise ValueError("KMeansSIDBuilder requires `embeddings` (np.ndarray).")
        if len(embeddings) != len(item_ids):
            raise ValueError(
                f"embeddings length ({len(embeddings)}) must match "
                f"item_ids length ({len(item_ids)})."
            )

        n = len(item_ids)
        k = min(self.vocab_size_per_token, n)

        sid_assignments = np.zeros((n, self.num_sid_tokens), dtype=np.int32)
        self.cluster_models = []

        for level in range(self.num_sid_tokens):
            # Use a different random state per level for diversity
            km = KMeans(
                n_clusters=k,
                max_iter=self.kmeans_iters,
                n_init=self.n_init,
                random_state=(self.seed if self.seed is not None else 42) + level,
            )
            cluster_ids = km.fit_predict(embeddings)
            sid_assignments[:, level] = cluster_ids
            self.cluster_models.append(km)

        sid_list = [tuple(row) for row in sid_assignments]
        return self._finalize(item_ids, sid_list)


class RQKMeansSIDBuilder(BaseSIDBuilder):
    """Residual Quantized KMeans: multi-level SIDs via residual clustering.

    Level 0 clusters the raw embeddings.
    Level 1 clusters the residual (embedding - level0 centroid).
    Level 2 clusters the residual after level 0 + level 1, etc.

    This mirrors the ResidualVectorQuantizer in rq/models/rq.py.
    """

    def __init__(
        self,
        num_sid_tokens: int = 3,
        vocab_size_per_token: int = 256,
        seed: Optional[int] = None,
        kmeans_iters: int = 100,
        n_init: int = 10,
    ):
        super().__init__(num_sid_tokens, vocab_size_per_token, seed)
        self.kmeans_iters = kmeans_iters
        self.n_init = n_init
        self.codebooks: List[np.ndarray] = []

    def build(
        self,
        item_ids: List[Any],
        embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[Dict[Any, Tuple[int, ...]], Dict[Tuple[int, ...], List[Any]]]:
        if embeddings is None:
            raise ValueError("RQKMeansSIDBuilder requires `embeddings` (np.ndarray).")
        if len(embeddings) != len(item_ids):
            raise ValueError(
                f"embeddings length ({len(embeddings)}) must match "
                f"item_ids length ({len(item_ids)})."
            )

        n = len(item_ids)
        k = min(self.vocab_size_per_token, n)
        residual = embeddings.copy().astype(np.float64)
        sid_assignments = np.zeros((n, self.num_sid_tokens), dtype=np.int32)
        self.codebooks = []

        for level in range(self.num_sid_tokens):
            km = KMeans(
                n_clusters=k,
                max_iter=self.kmeans_iters,
                n_init=self.n_init,
                random_state=(self.seed if self.seed is not None else 42) + level,
            )
            cluster_ids = km.fit_predict(residual)
            centroids = km.cluster_centers_.astype(np.float64)

            sid_assignments[:, level] = cluster_ids
            self.codebooks.append(centroids)

            # Subtract chosen centroid from residual for next level
            residual = residual - centroids[cluster_ids]

        sid_list = [tuple(row) for row in sid_assignments]
        return self._finalize(item_ids, sid_list)


def get_sid_builder(
    method: str,
    num_sid_tokens: int = 3,
    vocab_size_per_token: int = 256,
    seed: Optional[int] = None,
    **kwargs,
) -> BaseSIDBuilder:
    """Factory function: returns a SID builder by name."""
    builders = {
        "random": RandomSIDBuilder,
        "category": CategoryAwareSIDBuilder,
        "kmeans": KMeansSIDBuilder,
        "rqkmeans": RQKMeansSIDBuilder,
    }
    cls = builders.get(method.lower())
    if cls is None:
        raise ValueError(
            f"Unknown SID builder '{method}'. Choose from: {list(builders.keys())}"
        )
    return cls(num_sid_tokens=num_sid_tokens,
               vocab_size_per_token=vocab_size_per_token,
               seed=seed, **kwargs)
