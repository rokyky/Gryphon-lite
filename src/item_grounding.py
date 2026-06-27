"""
SID到物品候选的Grounding（任务3.1）。

给定生成的SID Token，查找候选物品，处理冲突，
并在冲突组内通过启发式方法排序（流行度、新近度）。
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ItemGrounding:
    """将生成的SID Token映射到具体的物品候选项。

    通过使用启发式分数对每个冲突组内的物品进行排名来处理SID冲突
    （多个物品映射到同一个SID）。

    Args:
        sid_to_items: 从SID元组到物品ID列表的映射。
        item_to_sid: 从物品ID到SID元组的映射。
        popularity_scores: 可选的物品ID到流行度分数的字典。
        recency_scores: 可选的物品ID到新近度分数的字典。
    """

    def __init__(
        self,
        sid_to_items: Dict[Tuple[int, ...], List[Any]],
        item_to_sid: Optional[Dict[Any, Tuple[int, ...]]] = None,
        popularity_scores: Optional[Dict[Any, float]] = None,
        recency_scores: Optional[Dict[Any, float]] = None,
    ):
        self.sid_to_items = sid_to_items
        self.item_to_sid = item_to_sid or {}
        self.popularity_scores = popularity_scores or {}
        self.recency_scores = recency_scores or {}

    def ground_sid(
        self,
        sid: Tuple[int, ...],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """查找生成的SID对应的候选物品。

        返回：
            (candidate_items, metadata):
                candidate_items: 映射到此SID的物品ID列表。
                metadata: 包含冲突信息的字典。
        """
        items = self.sid_to_items.get(sid, [])

        metadata = {
            "sid": sid,
            "num_candidates": len(items),
            "has_collision": len(items) > 1,
        }

        if len(items) <= 1:
            return items, metadata

        # 多个物品共享此SID；按启发式排序
        scored = []
        for item_id in items:
            score = self._heuristic_score(item_id)
            scored.append((item_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_items = [item_id for item_id, _ in scored]

        metadata["collision_scores"] = {
            str(item_id): score for item_id, score in scored
        }

        return ranked_items, metadata

    def ground_sids(
        self,
        sids: List[Tuple[int, ...]],
    ) -> Tuple[List[Tuple[Any, float]], Dict[str, Any]]:
        """将多个生成的SID映射到排序后的候选物品。

        Args:
            sids: 生成的SID元组列表。

        Returns:
            (ranked_candidates, metadata):
                ranked_candidates: (item_id, grounding_score)对列表，
                                   按分数降序排列。
                metadata: 包含grounding统计信息的字典。
        """
        all_candidates: List[Tuple[Any, float]] = []
        seen_items: set = set()
        grounding_metadata = {
            "total_sids": len(sids),
            "valid_sids": 0,
            "total_candidates_before_dedup": 0,
        }

        for sid in sids:
            items, meta = self.ground_sid(sid)

            if items:
                grounding_metadata["valid_sids"] += 1
                grounding_metadata["total_candidates_before_dedup"] += len(items)

            for item_id in items:
                if item_id not in seen_items:
                    score = self._heuristic_score(item_id)
                    all_candidates.append((item_id, score))
                    seen_items.add(item_id)

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        grounding_metadata["unique_candidates"] = len(all_candidates)

        return all_candidates, grounding_metadata

    def _heuristic_score(self, item_id: Any) -> float:
        """计算物品的启发式排名分数。

        结合流行度、新近度和一个小的默认分数。
        所有分量大致归一化到[0, 1]。
        """
        score = 1.0  # 默认

        if item_id in self.popularity_scores:
            score += self.popularity_scores[item_id]

        if item_id in self.recency_scores:
            score += self.recency_scores[item_id]

        return score

    def ground_with_item_embeddings(
        self,
        sids: List[Tuple[int, ...]],
        user_embedding: Optional[Any] = None,
        item_embeddings: Optional[Dict[Any, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """使用物品嵌入进行更好的冲突解决的SID Grounding。

        如果用户嵌入可用，则在每个冲突组内通过用户相似度对物品评分。

        Args:
            sids: 生成的SID元组列表。
            user_embedding: 可选的用户表示向量。
            item_embeddings: 可选的物品ID到嵌入向量的字典。

        Returns:
            按分数降序排列的(item_id, score)列表。
        """
        import numpy as np

        candidates: List[Tuple[Any, float]] = []
        seen_items: set = set()

        for sid in sids:
            items = self.sid_to_items.get(sid, [])
            for item_id in items:
                if item_id not in seen_items:
                    seen_items.add(item_id)

                    score = 1.0
                    # 如果可用，使用嵌入相似度
                    if (
                        user_embedding is not None
                        and item_embeddings is not None
                        and item_id in item_embeddings
                    ):
                        ue = np.asarray(user_embedding).flatten()
                        ie = np.asarray(item_embeddings[item_id]).flatten()
                        if ue.shape == ie.shape and np.linalg.norm(ue) > 0 and np.linalg.norm(ie) > 0:
                            sim = float(np.dot(ue, ie) / (np.linalg.norm(ue) * np.linalg.norm(ie)))
                            score += max(0.0, sim)

                    # 添加流行度加分
                    if item_id in self.popularity_scores:
                        score += self.popularity_scores[item_id]

                    candidates.append((item_id, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
