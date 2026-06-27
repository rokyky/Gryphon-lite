"""
SID mapping utilities: export, load, lookup, and trie construction for
constrained decoding.

Supports:
    - export_mappings: save item_to_sid and sid_to_items to JSON
    - load_mappings: restore from JSON
    - item_to_sid: O(1) lookup
    - sid_to_items: O(1) reverse lookup (handles collisions)
    - build_sid_trie: build prefix trie for constrained beam search
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
    """Save item-to-SID and SID-to-item mappings to a JSON file.

    Keys and values are converted to strings/ints for JSON serialization.
    SID tuples are stored as lists.

    If sid_to_items is not provided, it is reconstructed from item_to_sid.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if sid_to_items is None:
        sid_to_items = defaultdict(list)
        for item_id, sid in item_to_sid.items():
            sid_to_items[sid].append(item_id)
        sid_to_items = dict(sid_to_items)

    # Convert to JSON-serializable structures
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
    """Load item-to-SID and SID-to-item mappings from a JSON file.

    Returns:
        (item_to_sid, sid_to_items) dictionaries.
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

    # If sid_to_items was not stored, reconstruct it
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
    """Lookup the SID for a given item ID."""
    return item_to_sid.get(item_id)


def sid_to_items_lookup(
    sid: Union[Tuple[int, ...], List[int]],
    sid_to_items: Dict[Tuple[int, ...], List[Any]],
) -> List[Any]:
    """Reverse lookup: return all items that share this SID.

    Handles collisions: multiple items may map to the same SID.
    Returns an empty list if the SID is not found.
    """
    key = tuple(sid)
    return sid_to_items.get(key, [])


class SIDTrieNode:
    """A node in the SID trie for constrained decoding."""

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
    """Prefix trie over SID token sequences.

    Used for constrained decoding: at each generation step, only tokens
    that lead to a valid complete SID are allowed.

    Usage:
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
        """Insert a single SID tuple into the trie."""
        node = self.root
        for depth, token in enumerate(sid):
            if token not in node.children:
                node.children[token] = SIDTrieNode(depth=depth + 1)
            node = node.children[token]
        node.is_terminal = True
        self._num_sids += 1
        self._max_depth = max(self._max_depth, len(sid))

    def add_many(self, sids) -> None:
        """Insert multiple SID tuples."""
        for sid in sids:
            self.add(sid)

    def valid_next_tokens(self, prefix: Tuple[int, ...]) -> List[int]:
        """Return all valid token ids that can follow the given prefix.

        Returns an empty list if the prefix is invalid (not present in trie).
        """
        node = self._traverse(prefix)
        if node is None:
            return []
        return list(node.children.keys())

    def is_valid_prefix(self, prefix: Tuple[int, ...]) -> bool:
        """Check whether the given prefix exists in the trie."""
        return self._traverse(prefix) is not None

    def is_complete_sid(self, sid: Tuple[int, ...]) -> bool:
        """Check whether the given SID is a complete (terminal) entry."""
        node = self._traverse(sid)
        return node is not None and node.is_terminal

    def all_sids(self) -> List[Tuple[int, ...]]:
        """Return all complete SID sequences stored in the trie."""
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
        """Follow the prefix and return the final node, or None."""
        node = self.root
        for token in prefix:
            if token not in node.children:
                return None
            node = node.children[token]
        return node

    def __len__(self) -> int:
        return self._num_sids


class SIDMapper:
    """Unified mapper class for item-to-SID and SID-to-item lookups.

    Provides convenient lookup methods and wraps export/load functionality.

    Args:
        item_to_sid: mapping from item ID to SID tuple.
        sid_to_items: reverse mapping from SID tuple to list of item IDs.
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
        """Lookup the SID for a given item ID."""
        return self.item_to_sid.get(item_id)

    def sid_to_items_lookup(self, sid: Union[Tuple[int, ...], List[int]]) -> List[Any]:
        """Reverse lookup: return all items that share this SID.

        Handles collisions: multiple items may map to the same SID.
        Returns an empty list if the SID is not found.
        """
        key = tuple(sid)
        return self.sid_to_items.get(key, [])

    def export(self, output_path: str = "sid_mappings.json"):
        """Export mappings to JSON."""
        export_mappings(self.item_to_sid, self.sid_to_items, output_path)

    @classmethod
    def load(cls, path: str) -> "SIDMapper":
        """Load mappings from JSON and return an SIDMapper instance."""
        item_to_sid, sid_to_items = load_mappings(path)
        return cls(item_to_sid, sid_to_items)

    def build_trie(self) -> SIDTrie:
        """Build a SID trie from the current mappings."""
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
    """Build an SID trie from a sid_to_items mapping for constrained decoding."""
    trie = SIDTrie()
    for sid in sid_to_items:
        trie.add(sid)
    logger.info(f"Built SID trie with {trie.num_sids} entries, depth={trie.max_depth}")
    return trie
