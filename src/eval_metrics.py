"""
Recommendation evaluation metrics (Task 5.2).

Supports:
    - HR@K (Hit Rate)
    - NDCG@K (Normalized Discounted Cumulative Gain)
    - Recall@K
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

logger = logging.getLogger(__name__)


def hr_at_k(
    ranked_list: List[Any],
    ground_truth: List[Any],
    k: Optional[int] = None,
) -> float:
    """Compute Hit Rate @ K.

    HR@K = 1 if any ground truth item appears in top-K, else 0.

    Args:
        ranked_list: ranked list of item IDs (or candidate IDs).
        ground_truth: list of relevant (ground truth) item IDs.
        k: cutoff (default: len(ranked_list)).

    Returns:
        1.0 if hit, else 0.0.
    """
    if k is None:
        k = len(ranked_list)
    if k <= 0 or not ranked_list or not ground_truth:
        return 0.0

    top_k = set(ranked_list[:k])
    gt_set = set(ground_truth)

    return 1.0 if top_k & gt_set else 0.0


def ndcg_at_k(
    ranked_list: List[Any],
    ground_truth: List[Any],
    k: Optional[int] = None,
) -> float:
    """Compute NDCG @ K.

    NDCG@K = DCG@K / IDCG@K, where:
        DCG@K = sum(1 / log2(i + 1) for i where ranked_list[i] in ground_truth)
        IDCG@K = sum(1 / log2(i + 1) for i in range(min(K, |GT|)))

    Args:
        ranked_list: ranked list of item IDs.
        ground_truth: list of relevant item IDs.
        k: cutoff (default: len(ranked_list)).

    Returns:
        NDCG value in [0, 1].
    """
    if k is None:
        k = len(ranked_list)
    if k <= 0 or not ranked_list or not ground_truth:
        return 0.0

    gt_set = set(ground_truth)
    k = min(k, len(ranked_list))

    dcg = 0.0
    for i in range(k):
        if ranked_list[i] in gt_set:
            dcg += 1.0 / np.log2(i + 2)

    # Ideal DCG
    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    ranked_list: List[Any],
    ground_truth: List[Any],
    k: Optional[int] = None,
) -> float:
    """Compute Recall @ K.

    Recall@K = |top-K intersect GT| / |GT|

    Args:
        ranked_list: ranked list of item IDs.
        ground_truth: list of relevant item IDs.
        k: cutoff (default: len(ranked_list)).

    Returns:
        Recall value in [0, 1].
    """
    if k is None:
        k = len(ranked_list)
    if k <= 0 or not ranked_list or not ground_truth:
        return 0.0

    top_k = set(ranked_list[:k])
    gt_set = set(ground_truth)

    if not gt_set:
        return 0.0

    hits = len(top_k & gt_set)
    return hits / len(gt_set)


def mean_metric_at_ks(
    scores_by_user: Dict[Any, List[Any]],
    ground_truth_by_user: Dict[Any, List[Any]],
    ks: List[int],
    metric_fn,
) -> Dict[int, float]:
    """Compute mean metric @ K over all users.

    Args:
        scores_by_user: dict of user_id -> ranked list of item IDs.
        ground_truth_by_user: dict of user_id -> list of relevant item IDs.
        ks: list of K values to evaluate.
        metric_fn: metric function (hr_at_k, ndcg_at_k, recall_at_k).

    Returns:
        Dict mapping K -> mean metric value.
    """
    results = {}
    for k in ks:
        per_user = []
        for user_id in scores_by_user:
            if user_id not in ground_truth_by_user:
                continue
            val = metric_fn(scores_by_user[user_id], ground_truth_by_user[user_id], k)
            per_user.append(val)
        results[k] = float(np.mean(per_user)) if per_user else 0.0
    return results


def evaluate_full_ranking(
    scores_by_user: Dict[Any, List[Any]],
    ground_truth_by_user: Dict[Any, List[Any]],
    ks: Optional[List[int]] = None,
) -> Dict[str, Dict[int, float]]:
    """Full ranking evaluation with HR, NDCG, and Recall.

    Args:
        scores_by_user: dict of user_id -> ranked list of item IDs.
        ground_truth_by_user: dict of user_id -> list of relevant item IDs.
        ks: list of K values (default: [1, 5, 10, 20]).

    Returns:
        Dict with keys "HR", "NDCG", "Recall", each mapping K -> value.
    """
    if ks is None:
        ks = [1, 5, 10, 20]

    return {
        "HR": mean_metric_at_ks(scores_by_user, ground_truth_by_user, ks, hr_at_k),
        "NDCG": mean_metric_at_ks(scores_by_user, ground_truth_by_user, ks, ndcg_at_k),
        "Recall": mean_metric_at_ks(scores_by_user, ground_truth_by_user, ks, recall_at_k),
    }
