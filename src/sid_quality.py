"""
用于评估Semantic ID分配和生成SID质量的SID质量指标。

任务1.4指标（SID质量）：
    - collision_rate（冲突率）
    - code_utilization（代码利用率）
    - category_purity（类别纯度）
    - collision_group_stats（冲突组统计）

任务2.5指标（生成质量）：
    - valid_sid_rate（有效SID率）
    - valid_item_rate（有效物品率）
    - duplicate_rate（重复率）
    - beam_diversity（Beam多样性）
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.sid_mapper import SIDTrie

logger = logging.getLogger(__name__)


# ===== 任务1.4：SID分配质量 =====


def collision_rate(
    item_to_sid: Dict[Any, Tuple[int, ...]],
    sid_to_items: Optional[Dict[Tuple[int, ...], List[Any]]] = None,
) -> float:
    """与至少一个其他物品共享其SID的物品所占的比例。

    如果没有物品，返回0.0。
    """
    total = len(item_to_sid)
    if total == 0:
        return 0.0

    if sid_to_items is None:
        sid_to_items = defaultdict(list)
        for item_id, sid in item_to_sid.items():
            sid_to_items[sid].append(item_id)

    colliding = sum(1 for items in sid_to_items.values() if len(items) > 1)
    unique_sids = len(sid_to_items)
    return colliding / max(unique_sids, 1)


def code_utilization(
    sid_to_items: Dict[Tuple[int, ...], List[Any]],
    vocab_size_per_token: int = 256,
) -> Dict[str, float]:
    """实际使用的可能代码所占的比例。

    返回每级利用率和完整路径利用率。
    """
    if not sid_to_items:
        return {}

    num_levels = len(next(iter(sid_to_items)))
    result: Dict[str, float] = {}

    for level in range(num_levels):
        used = len(set(sid[level] for sid in sid_to_items))
        result[f"level_{level}"] = used / max(vocab_size_per_token, 1)

    total_possible = vocab_size_per_token ** num_levels
    result["full_path"] = len(sid_to_items) / max(total_possible, 1)
    return result


def category_purity(
    sid_to_items: Dict[Tuple[int, ...], List[Any]],
    item_to_category: Dict[Any, str],
) -> float:
    """对于每个SID，共享相同多数类别物品的比例。

    在所有SID上加权平均。
    """
    if not sid_to_items:
        return 0.0

    total_weight = 0.0
    purity_sum = 0.0

    for sid, items in sid_to_items.items():
        category_counts: Counter = Counter()
        for item_id in items:
            cat = item_to_category.get(item_id)
            if cat is not None:
                category_counts[cat] += 1

        if category_counts and items:
            majority_count = max(category_counts.values())
            purity = majority_count / len(items)
            purity_sum += purity * len(items)
            total_weight += len(items)

    return purity_sum / max(total_weight, 1.0)


def collision_group_stats(
    sid_to_items: Dict[Tuple[int, ...], List[Any]],
) -> Dict[str, Any]:
    """冲突组大小的分布。

    返回：
        {
            "num_groups": int,       （组数）
            "max_size": int,         （最大组大小）
            "mean_size": float,      （平均大小）
            "median_size": float,    （中位数大小）
            "size_distribution": {size: count, ...}  （大小分布）
        }
    """
    group_sizes = [len(items) for items in sid_to_items.values() if len(items) > 1]

    if not group_sizes:
        return {
            "num_groups": 0,
            "max_size": 0,
            "mean_size": 0.0,
            "median_size": 0.0,
            "size_distribution": {},
        }

    sizes_sorted = sorted(group_sizes)
    n = len(sizes_sorted)

    return {
        "num_groups": n,
        "max_size": max(sizes_sorted),
        "mean_size": float(np.mean(sizes_sorted)),
        "median_size": float(sizes_sorted[n // 2]),
        "size_distribution": dict(Counter(sizes_sorted)),
    }


# ===== 任务2.5：生成质量指标 =====


def valid_sid_rate(
    generated_sids: List[Tuple[int, ...]],
    trie: SIDTrie,
) -> float:
    """生成的SID中存在于目录Trie中的比例。"""
    if not generated_sids:
        return 0.0

    valid = sum(1 for sid in generated_sids if trie.is_complete_sid(sid))
    return valid / len(generated_sids)


def valid_item_rate(
    generated_sids: List[Tuple[int, ...]],
    sid_to_items: Dict[Tuple[int, ...], List[Any]],
) -> float:
    """生成的SID中映射到至少一个真实物品的比例。"""
    if not generated_sids:
        return 0.0

    valid = sum(1 for sid in generated_sids if sid in sid_to_items)
    return valid / len(generated_sids)


def duplicate_rate(
    generated_item_ids: List[Any],
) -> float:
    """生成列表中重复物品的比例。"""
    if not generated_item_ids:
        return 0.0

    num_duplicates = len(generated_item_ids) - len(set(generated_item_ids))
    return num_duplicates / len(generated_item_ids)


def beam_diversity(
    generated_item_ids: List[Any],
) -> float:
    """唯一物品数除以Beam大小。"""
    if not generated_item_ids:
        return 0.0

    unique = len(set(generated_item_ids))
    return unique / len(generated_item_ids)


def compute_generation_metrics(
    generated_sids: List[Tuple[int, ...]],
    generated_item_ids: List[Any],
    trie: SIDTrie,
    sid_to_items: Dict[Tuple[int, ...], List[Any]],
) -> Dict[str, float]:
    """一次性计算所有生成质量指标。

    Args:
        generated_sids: 来自beam search的SID元组列表。
        generated_item_ids: 从生成的SID中grounding得到的物品ID。
        trie: 从目录构建的SID Trie。
        sid_to_items: 从SID到物品ID的反向映射。

    Returns:
        指标名称到值的字典。
    """
    return {
        "valid_sid_rate": valid_sid_rate(generated_sids, trie),
        "valid_item_rate": valid_item_rate(generated_sids, sid_to_items),
        "duplicate_rate": duplicate_rate(generated_item_ids),
        "beam_diversity": beam_diversity(generated_item_ids),
    }
