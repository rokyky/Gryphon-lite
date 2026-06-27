"""
SID quality metrics for evaluating Semantic ID assignments and generated SIDs.

Task 1.4 metrics (SID quality):
    - collision_rate
    - code_utilization
    - category_purity
    - collision_group_stats

Task 2.5 metrics (generation quality):
    - valid_sid_rate
    - valid_item_rate
    - duplicate_rate
    - beam_diversity
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.sid_mapper import SIDTrie

logger = logging.getLogger(__name__)


# ===== Task 1.4: SID assignment quality =====


def collision_rate(
    item_to_sid: Dict[Any, Tuple[int, ...]],
    sid_to_items: Optional[Dict[Tuple[int, ...], List[Any]]] = None,
) -> float:
    """Fraction of items that share their SID with at least one other item.

    Returns 0.0 if there are no items.
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
    """Fraction of possible codes that are actually used.

    Returns per-level utilization and full-path utilization.
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
    """For each SID, what fraction of items share the same majority category.

    Weighted average across all SIDs.
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
    """Distribution of collision group sizes.

    Returns:
        {
            "num_groups": int,
            "max_size": int,
            "mean_size": float,
            "median_size": float,
            "size_distribution": {size: count, ...}
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


# ===== Task 2.5: Generation quality metrics =====


def valid_sid_rate(
    generated_sids: List[Tuple[int, ...]],
    trie: SIDTrie,
) -> float:
    """Fraction of generated SIDs that exist in the catalog trie."""
    if not generated_sids:
        return 0.0

    valid = sum(1 for sid in generated_sids if trie.is_complete_sid(sid))
    return valid / len(generated_sids)


def valid_item_rate(
    generated_sids: List[Tuple[int, ...]],
    sid_to_items: Dict[Tuple[int, ...], List[Any]],
) -> float:
    """Fraction of generated SIDs that map to at least one real item."""
    if not generated_sids:
        return 0.0

    valid = sum(1 for sid in generated_sids if sid in sid_to_items)
    return valid / len(generated_sids)


def duplicate_rate(
    generated_item_ids: List[Any],
) -> float:
    """Fraction of duplicate items in the generated list."""
    if not generated_item_ids:
        return 0.0

    num_duplicates = len(generated_item_ids) - len(set(generated_item_ids))
    return num_duplicates / len(generated_item_ids)


def beam_diversity(
    generated_item_ids: List[Any],
) -> float:
    """Unique items divided by beam size."""
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
    """Compute all generation quality metrics at once.

    Args:
        generated_sids: list of SID tuples from beam search.
        generated_item_ids: grounded item IDs from the generated SIDs.
        trie: SID trie built from the catalog.
        sid_to_items: reverse mapping from SID to item IDs.

    Returns:
        Dictionary of metric name -> value.
    """
    return {
        "valid_sid_rate": valid_sid_rate(generated_sids, trie),
        "valid_item_rate": valid_item_rate(generated_sids, sid_to_items),
        "duplicate_rate": duplicate_rate(generated_item_ids),
        "beam_diversity": beam_diversity(generated_item_ids),
    }
