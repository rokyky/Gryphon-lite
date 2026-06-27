"""
SID映射工具：导出、加载、查找以及用于约束解码的Trie构建。

支持：
    - export_mappings: 将item_to_sid和sid_to_items保存为JSON
    - load_mappings: 从JSON恢复
    - item_to_sid: O(1)查找
    - sid_to_items: O(1)反向查找（处理冲突）
    - build_sid_trie: 为约束beam search构建前缀Trie
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


def export_mappings(
    item_to_sid: Dict[Any, Tuple[int, ...]],
    sid_to_items: Optional[Dict[Tuple[int, ...], List[Any]]] = None,
    output_path: str = "sid_mappings.json",
):
    """将item-to-SID和SID-to-item映射保存到JSON文件。

    键和值会转换为字符串/整数以进行JSON序列化。
    SID元组存储为列表。

    如果未提供sid_to_items，则从item_to_sid重建。
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if sid_to_items is None:
        sid_to_items = defaultdict(list)
        for item_id, sid in item_to_sid.items():
            sid_to_items[sid].append(item_id)
        sid_to_items = dict(sid_to_items)

    # 转换为JSON可序列化的结构
    item_to_sid_serial = {}
    for item_id, sid in item_to_sid.items():
        item_to_sid_serial[str(item_id)] = [int(t) for t in sid]

    sid_to_items_serial = {}
    for sid, items in sid_to_items.items():
        sid_key = "-".join(str(t) for t in sid)
        sid_to_items_serial[sid_key] = [str(i) for i in items]

    data = {
        "item_to_sid": item_to_sid_serial,
        "sid_to_items": sid_to_items_serial,
        "metadata": {
            "num_items": len(item_to_sid),
            "num_unique_sids": len(sid_to_items),
            "collision_rate": (
                (len(item_to_sid) - len(sid_to_items)) / max(len(item_to_sid), 1)
            ),
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported {len(item_to_sid)} item->SID mappings to {output_path}")


def load_mappings(path: str) -> Tuple[Dict[Any, Tuple[int, ...]], Dict[Tuple[int, ...], List[Any]]]:
    """从JSON文件加载item-to-SID和SID-to-item映射。

    返回：
        (item_to_sid, sid_to_items)字典。
    """
    with open(path, "r") as f:
        data = json.load(f)

    item_to_sid = {}
    for item_id_str, sid_list in data["item_to_sid"].items():
        item_to_sid[item_id_str] = tuple(sid_list)

    sid_to_items: Dict[Tuple[int, ...], List[Any]] = {}
    for sid_key, items in data.get("sid_to_items", {}).items():
        sid = tuple(int(t) for t in sid_key.split("-"))
        sid_to_items[sid] = items

    # 如果sid_to_items未存储，则重建它
    if not sid_to_items:
        sid_to_items = defaultdict(list)
        for item_id, sid in item_to_sid.items():
            sid_to_items[sid].append(item_id)
        sid_to_items = dict(sid_to_items)

    logger.info(
        f"Loaded {len(item_to_sid)} item->SID mappings, "
        f"{len(sid_to_items)} unique SIDs from {path}"
    )
    return item_to_sid, sid_to_items


def item_to_sid_lookup(
    item_id: Any,
    item_to_sid: Dict[Any, Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    """查找给定物品ID对应的SID。"""
    return item_to_sid.get(item_id)


def sid_to_items_lookup(
    sid: Union[Tuple[int, ...], List[int]],
    sid_to_items: Dict[Tuple[int, ...], List[Any]],
) -> List[Any]:
    """反向查找：返回共享此SID的所有物品。

    处理冲突：多个物品可能映射到同一个SID。
    如果未找到SID，返回空列表。
    """
    key = tuple(sid)
    return sid_to_items.get(key, [])


class SIDTrieNode:
    """用于约束解码的SID Trie节点。"""

    __slots__ = ("children", "is_terminal", "depth")

    def __init__(self, depth: int = 0):
        self.children: Dict[int, "SIDTrieNode"] = {}
        self.is_terminal: bool = False
        self.depth: int = depth

    def __repr__(self):
        return (
            f"SIDTrieNode(depth={self.depth}, "
            f"children={len(self.children)}, "
            f"terminal={self.is_terminal})"
        )


class SIDTrie:
    """SID Token序列的前缀Trie。

    用于约束解码：在每个生成步骤中，只允许那些能导向有效完整SID的Token。

    用法：
        trie = SIDTrie()
        trie.add((1, 42, 7))
        trie.add((1, 42, 8))
        valid_next = trie.valid_next_tokens((1,))   # -> [42]
        trie.is_valid_prefix((1, 42))                # -> True
        trie.is_complete_sid((1, 42, 7))             # -> True
    """

    def __init__(self):
        self.root = SIDTrieNode(depth=0)
        self._num_sids = 0
        self._max_depth = 0

    def add(self, sid: Tuple[int, ...]) -> None:
        """将单个SID元组插入Trie中。"""
        node = self.root
        for depth, token in enumerate(sid):
            if token not in node.children:
                node.children[token] = SIDTrieNode(depth=depth + 1)
            node = node.children[token]
        node.is_terminal = True
        self._num_sids += 1
        self._max_depth = max(self._max_depth, len(sid))

    def add_many(self, sids) -> None:
        """插入多个SID元组。"""
        for sid in sids:
            self.add(sid)

    def valid_next_tokens(self, prefix: Tuple[int, ...]) -> List[int]:
        """返回所有可以跟在给定前缀后面的有效TokenID。

        如果前缀无效（不在Trie中），返回空列表。
        """
        node = self._traverse(prefix)
        if node is None:
            return []
        return list(node.children.keys())

    def is_valid_prefix(self, prefix: Tuple[int, ...]) -> bool:
        """检查给定的前缀是否存在于Trie中。"""
        return self._traverse(prefix) is not None

    def is_complete_sid(self, sid: Tuple[int, ...]) -> bool:
        """检查给定的SID是否是完整的（终端的）条目。"""
        node = self._traverse(sid)
        return node is not None and node.is_terminal

    def all_sids(self) -> List[Tuple[int, ...]]:
        """返回存储在Trie中的所有完整SID序列。"""
        result: List[Tuple[int, ...]] = []

        def _dfs(node: SIDTrieNode, path: List[int]):
            if node.is_terminal:
                result.append(tuple(path))
            for token, child in node.children.items():
                path.append(token)
                _dfs(child, path)
                path.pop()

        _dfs(self.root, [])
        return result

    @property
    def num_sids(self) -> int:
        return self._num_sids

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def _traverse(self, prefix: Tuple[int, ...]) -> Optional[SIDTrieNode]:
        """沿着前缀遍历并返回最终节点，如果不存在则返回None。"""
        node = self.root
        for token in prefix:
            if token not in node.children:
                return None
            node = node.children[token]
        return node

    def __len__(self) -> int:
        return self._num_sids


class SIDMapper:
    """统一的映射器类，用于item-to-SID和SID-to-item查找。

    提供便捷的查找方法并封装了导出/加载功能。

    Args:
        item_to_sid: 从物品ID到SID元组的映射。
        sid_to_items: 从SID元组到物品ID列表的反向映射。
    """

    def __init__(
        self,
        item_to_sid: Dict[Any, Tuple[int, ...]],
        sid_to_items: Optional[Dict[Tuple[int, ...], List[Any]]] = None,
    ):
        self.item_to_sid = item_to_sid
        if sid_to_items is None:
            self.sid_to_items = defaultdict(list)
            for item_id, sid in item_to_sid.items():
                self.sid_to_items[sid].append(item_id)
            self.sid_to_items = dict(self.sid_to_items)
        else:
            self.sid_to_items = sid_to_items

    def item_to_sid_lookup(self, item_id: Any) -> Optional[Tuple[int, ...]]:
        """查找给定物品ID对应的SID。"""
        return self.item_to_sid.get(item_id)

    def sid_to_items_lookup(self, sid: Union[Tuple[int, ...], List[int]]) -> List[Any]:
        """反向查找：返回共享此SID的所有物品。

        处理冲突：多个物品可能映射到同一个SID。
        如果未找到SID，返回空列表。
        """
        key = tuple(sid)
        return self.sid_to_items.get(key, [])

    def export(self, output_path: str = "sid_mappings.json"):
        """将映射导出到JSON。"""
        export_mappings(self.item_to_sid, self.sid_to_items, output_path)

    @classmethod
    def load(cls, path: str) -> "SIDMapper":
        """从JSON加载映射并返回SIDMapper实例。"""
        item_to_sid, sid_to_items = load_mappings(path)
        return cls(item_to_sid, sid_to_items)

    def build_trie(self) -> SIDTrie:
        """从当前映射构建一个SID Trie。"""
        return build_sid_trie(self.sid_to_items)

    @property
    def num_items(self) -> int:
        return len(self.item_to_sid)

    @property
    def num_unique_sids(self) -> int:
        return len(self.sid_to_items)

    @property
    def collision_rate(self) -> float:
        if self.num_items == 0:
            return 0.0
        return (self.num_items - self.num_unique_sids) / self.num_items


def build_sid_trie(
    sid_to_items: Dict[Tuple[int, ...], List[Any]],
) -> SIDTrie:
    """从sid_to_items映射构建用于约束解码的SID Trie。"""
    trie = SIDTrie()
    for sid in sid_to_items:
        trie.add(sid)
    logger.info(f"Built SID trie with {trie.num_sids} entries, depth={trie.max_depth}")
    return trie
