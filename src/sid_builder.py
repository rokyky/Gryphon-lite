"""
用于将物品映射到Token序列的Semantic ID（SID）构建器。

支持多种构建策略：
    - RandomSIDBuilder：分配随机Token序列作为SID
    - CategoryAwareSIDBuilder：前缀为类别ID + 随机后缀
    - KMeansSIDBuilder：对文本嵌入进行聚类，使用聚类ID作为SID Token
    - RQKMeansSIDBuilder：残差量化KMeans，用于多级SID

所有构建器都提供：
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
    """所有SID构建器的抽象基类。"""

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
        """为给定的物品ID构建SID映射。"""
        ...

    def _finalize(self, item_ids: List[Any], sid_assignments: List[Tuple[int, ...]]):
        """从并行列表中填充item_to_sid和sid_to_items。"""
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
    """分配均匀随机Token序列作为SID。"""

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
    """前缀为类别ID + 随机后缀作为SID。

    需要`categories`关键字参数：与item_ids对应的类别标签列表。
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

        # 如果未提供类别词汇表，则构建一个
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
    """通过KMeans对文本嵌入进行聚类，并使用聚类ID作为SID Token。

    这产生扁平聚类：每个SID Token级别来自对相同嵌入的独立KMeans运行，
    因此不同位置的Token捕获互补的聚类结构。
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
            # 为每个级别使用不同的随机状态以增加多样性
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
    """残差量化KMeans：通过残差聚类实现多级SID。

    第0级对原始嵌入进行聚类。
    第1级对残差（嵌入 - 第0级质心）进行聚类。
    第2级对减去第0级和第1级后的残差进行聚类，以此类推。

    这对应于rq/models/rq.py中的ResidualVectorQuantizer。
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

            # 从残差中减去选定的质心，用于下一级
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
    """工厂函数：根据名称返回SID构建器。"""
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
