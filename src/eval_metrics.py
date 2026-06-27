"""
推荐评估指标（任务5.2）。

支持：
    - HR@K（命中率）
    - NDCG@K（归一化折损累计增益）
    - Recall@K（召回率）
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
    """计算Hit Rate @ K。

    HR@K = 如果任何真实物品出现在top-K中则为1，否则为0。

    Args:
        ranked_list: 排序后的物品ID列表（或候选ID）。
        ground_truth: 相关（真实）物品ID列表。
        k: 截断值（默认：len(ranked_list)）。

    Returns:
        命中返回1.0，否则返回0.0。
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
    """计算NDCG @ K。

    NDCG@K = DCG@K / IDCG@K，其中：
        DCG@K = sum(1 / log2(i + 1) for i where ranked_list[i] in ground_truth)
        IDCG@K = sum(1 / log2(i + 1) for i in range(min(K, |GT|)))

    Args:
        ranked_list: 排序后的物品ID列表。
        ground_truth: 相关物品ID列表。
        k: 截断值（默认：len(ranked_list)）。

    Returns:
        NDCG值，范围[0, 1]。
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

    # 理想DCG
    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    ranked_list: List[Any],
    ground_truth: List[Any],
    k: Optional[int] = None,
) -> float:
    """计算Recall @ K。

    Recall@K = |top-K与真实集交集| / |真实集|

    Args:
        ranked_list: 排序后的物品ID列表。
        ground_truth: 相关物品ID列表。
        k: 截断值（默认：len(ranked_list)）。

    Returns:
        Recall值，范围[0, 1]。
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
    """计算所有用户在K处的平均指标。

    Args:
        scores_by_user: 用户ID到排序后物品ID列表的字典。
        ground_truth_by_user: 用户ID到相关物品ID列表的字典。
        ks: 要评估的K值列表。
        metric_fn: 指标函数（hr_at_k, ndcg_at_k, recall_at_k）。

    Returns:
        映射K到平均指标值的字典。
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
    """使用HR、NDCG和Recall的完整排名评估。

    Args:
        scores_by_user: 用户ID到排序后物品ID列表的字典。
        ground_truth_by_user: 用户ID到相关物品ID列表的字典。
        ks: K值列表（默认：[1, 5, 10, 20]）。

    Returns:
        包含"HR"、"NDCG"、"Recall"键的字典，每个映射K到值。
    """
    if ks is None:
        ks = [1, 5, 10, 20]

    return {
        "HR": mean_metric_at_ks(scores_by_user, ground_truth_by_user, ks, hr_at_k),
        "NDCG": mean_metric_at_ks(scores_by_user, ground_truth_by_user, ks, ndcg_at_k),
        "Recall": mean_metric_at_ks(scores_by_user, ground_truth_by_user, ks, recall_at_k),
    }
