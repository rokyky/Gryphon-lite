"""
SID元数据追踪：冲突组、前缀统计、代码利用率。

提供分析和导出已构建SID映射统计信息的工具。
"""

import json
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SIDMetadataTracker:
    """追踪和计算SID分配相关的元数据。

    Args:
        item_to_sid: 从物品ID到SID元组的映射。
        sid_to_items: 从SID元组到物品ID列表的反向映射。
    """

    def __init__(
        self,
        item_to_sid: Dict[Any, Tuple[int, ...]],
        sid_to_items: Dict[Tuple[int, ...], List[Any]],
    ):
        self.item_to_sid = item_to_sid
        self.sid_to_items = sid_to_items
        self._collision_groups: Optional[List[List[Any]]] = None
        self._prefix_stats: Optional[Dict[str, int]] = None
        self._code_utilization: Optional[Dict[str, float]] = None

    # ----- collision groups -----

    @property
    def collision_groups(self) -> List[List[Any]]:
        """共享相同完整SID的物品组。

        每组大小 >= 2。
        """
        if self._collision_groups is None:
            self._collision_groups = [
                items for items in self.sid_to_items.values() if len(items) > 1
            ]
        return self._collision_groups

    @property
    def num_collision_groups(self) -> int:
        return len(self.collision_groups)

    @property
    def total_colliding_items(self) -> int:
        """与至少一个其他物品共享其SID的物品总数。"""
        return sum(len(g) for g in self.collision_groups)

    @property
    def collision_group_size_distribution(self) -> Dict[int, int]:
        """从组大小到该大小组数的映射。"""
        sizes = Counter(len(g) for g in self.collision_groups)
        return dict(sorted(sizes.items()))

    # ----- prefix statistics -----

    @property
    def prefix_stats(self) -> Dict[str, int]:
        """统计每个前缀（前K个Token）被多少物品共享。

        键的格式为"depth=K:token1-token2-..."。
        """
        if self._prefix_stats is None:
            self._prefix_stats = {}
            for sid in self.sid_to_items:
                for depth in range(1, len(sid) + 1):
                    prefix = sid[:depth]
                    key = f"depth={depth}:" + "-".join(str(t) for t in prefix)
                    self._prefix_stats[key] = self._prefix_stats.get(key, 0) + 1
        return self._prefix_stats

    def get_prefix_collision_rate(self, depth: int) -> float:
        """在给定深度处，映射到超过1个SID的唯一前缀所占的比例。"""
        prefix_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
        for sid in self.sid_to_items:
            prefix = sid[:depth]
            prefix_counts[prefix] += 1

        if not prefix_counts:
            return 0.0
        colliding = sum(1 for c in prefix_counts.values() if c > 1)
        return colliding / len(prefix_counts)

    # ----- code utilization -----

    @property
    def code_utilization(self) -> Dict[str, float]:
        """每个深度级别实际使用的可能代码所占的比例。"""
        if self._code_utilization is None:
            if not self.item_to_sid:
                self._code_utilization = {}
                return self._code_utilization

            # 从实际值推断词汇表大小
            max_tokens_per_level: Dict[int, int] = defaultdict(int)
            for sid in self.sid_to_items:
                for level, token in enumerate(sid):
                    max_tokens_per_level[level] = max(max_tokens_per_level[level], token)

            num_levels = max(max_tokens_per_level.keys()) + 1 if max_tokens_per_level else 0
            util: Dict[str, float] = {}
            for level in range(num_levels):
                used = set()
                for sid in self.sid_to_items:
                    if level < len(sid):
                        used.add(sid[level])
                vocab_size = max_tokens_per_level[level] + 1
                util[f"level_{level}"] = len(used) / max(vocab_size, 1)

            # 完整路径利用率
            total_possible = 1
            for level in range(num_levels):
                total_possible *= (max_tokens_per_level[level] + 1)
            util["full_path"] = len(self.sid_to_items) / max(total_possible, 1)

            self._code_utilization = util

        return self._code_utilization

    # ----- export -----

    def export_metadata(self, output_path: str = "sid_metadata.json"):
        """将所有计算的元数据导出到JSON文件。"""
        metadata = {
            "num_items": len(self.item_to_sid),
            "num_unique_sids": len(self.sid_to_items),
            "collision_rate": (
                (len(self.item_to_sid) - len(self.sid_to_items))
                / max(len(self.item_to_sid), 1)
            ),
            "num_collision_groups": self.num_collision_groups,
            "total_colliding_items": self.total_colliding_items,
            "collision_group_size_distribution": self.collision_group_size_distribution,
            "code_utilization": self.code_utilization,
            "num_sid_tokens": (
                len(next(iter(self.sid_to_items)))
                if self.sid_to_items else 0
            ),
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"SID metadata exported to {output_path}")
        return metadata
