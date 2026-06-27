"""
SID-to-item candidate grounding (Task 3.1).

Given generated SID tokens, looks up candidate items, handles collisions,
and ranks within collision groups by heuristic (popularity, recency).
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ItemGrounding:
    """Ground generated SID tokens to concrete item candidates.

    Handles SID collisions (multiple items mapping to the same SID) by
    ranking items within each collision group using heuristic scores.

    Args:
        sid_to_items: mapping from SID tuple to list of item IDs.
        item_to_sid: mapping from item ID to SID tuple.
        popularity_scores: optional dict of item_id -> popularity score.
        recency_scores: optional dict of item_id -> recency score.
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
        """Look up candidate items for a generated SID.

        Returns:
            (candidate_items, metadata):
                candidate_items: list of item IDs that map to this SID.
                metadata: dict with collision info.
        """
        items = self.sid_to_items.get(sid, [])

        metadata = {
            "sid": sid,
            "num_candidates": len(items),
            "has_collision": len(items) > 1,
        }

        if len(items) <= 1:
            return items, metadata

        # Multiple items share this SID; rank by heuristic
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
        """Ground multiple generated SIDs to ranked candidate items.

        Args:
            sids: list of generated SID tuples.

        Returns:
            (ranked_candidates, metadata):
                ranked_candidates: list of (item_id, grounding_score) pairs,
                                   sorted by score descending.
                metadata: dict with grounding stats.
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
        """Compute a heuristic ranking score for an item.

        Combines popularity, recency, and a small default score.
        All components are normalized to roughly [0, 1].
        """
        score = 1.0  # default

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
        """Ground SIDs using item embeddings for better collision resolution.

        If a user embedding is available, score items by similarity to user
        within each collision group.

        Args:
            sids: list of generated SID tuples.
            user_embedding: optional user representation vector.
            item_embeddings: optional dict of item_id -> embedding vector.

        Returns:
            List of (item_id, score) sorted by score descending.
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
                    # Use embedding similarity if available
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

                    # Add popularity boost
                    if item_id in self.popularity_scores:
                        score += self.popularity_scores[item_id]

                    candidates.append((item_id, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
