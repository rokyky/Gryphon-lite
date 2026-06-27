"""
Trie约束的beam search解码器，用于SID生成。

确保生成的SID序列根据从已知SID目录构建的前缀Trie是有效的（任务2.3）。

只探索能够导向至少一个有效完整SID的下一个Token。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.sid_mapper import SIDTrie  # noqa: F401 — re-export for convenience
from src.sid_generator import SIDGenerator

logger = logging.getLogger(__name__)


@dataclass
class TrieBeamSearchConfig:
    """Trie约束beam search的配置。

    Attributes:
        beam_width: 维持的beam数量。
        max_sid_length: 最大SID Token长度（默认：从模型获取）。
        length_penalty: 长度归一化的指数（1.0 = 中性）。
        temperature: Softmax温度。
    """
    beam_width: int = 10
    max_sid_length: int = 0  # 0 = auto from model
    length_penalty: float = 1.0
    temperature: float = 1.0


class TrieConstrainedBeamSearch:
    """受SID Trie约束的beam search。

    在每个步骤中，只考虑当前前缀在Trie中存在的Token。
    这保证所有输出的SID都是有效的目录条目。

    用法：
        trie = build_sid_trie(sid_to_items)
        decoder = TrieConstrainedBeamSearch(trie, config)
        sids, scores = decoder.search(history_sids, model)
    """

    def __init__(
        self,
        trie: SIDTrie,
        config: Optional[TrieBeamSearchConfig] = None,
    ):
        self.trie = trie
        self.config = config or TrieBeamSearchConfig()
        self.max_sid_length = self.config.max_sid_length or trie.max_depth

    @torch.no_grad()
    def search(
        self,
        history_sids: torch.Tensor,
        model: SIDGenerator,
        num_return: Optional[int] = None,
    ) -> Tuple[List[List[Tuple[int, ...]]], List[List[float]]]:
        """运行Trie约束的beam search。

        Args:
            history_sids: (batch, history_len, num_sid_tokens) 历史序列。
            model: SID生成器模型。
            num_return: 每个batch返回的顶部序列数量（默认：beam_width）。

        Returns:
            (sequences, scores):
                sequences: SID元组的列表的列表，每个batch项一个。
                scores: beam分数的列表的列表，每个batch项一个。
        """
        B = history_sids.shape[0]
        beam_width = self.config.beam_width
        max_len = self.max_sid_length
        num_return = num_return or beam_width
        device = history_sids.device

        all_batch_sequences: List[List[Tuple[int, ...]]] = []
        all_batch_scores: List[List[float]] = []

        for batch_idx in range(B):
            single_history = history_sids[batch_idx:batch_idx + 1]  # (1, H, T)

            # 初始化beams：(prefix, score)
            beams: List[Tuple[Tuple[int, ...], float]] = [((), 0.0)]

            for step in range(max_len):
                new_beams: List[Tuple[Tuple[int, ...], float]] = []

                for prefix, score in beams:
                    # 从Trie获取有效的下一个Token
                    valid_tokens = self.trie.valid_next_tokens(prefix)
                    if not valid_tokens:
                        # 此beam无法继续；如果是一个完整SID则保留
                        if self.trie.is_complete_sid(prefix):
                            new_beams.append((prefix, score))
                        continue

                    # 如果此前缀已经是完整的SID，保持原样
                    if self.trie.is_complete_sid(prefix):
                        new_beams.append((prefix, score))
                        continue

                    # 获取模型对下一个Token的logits
                    token_logits = model.get_next_token_logits(
                        single_history, list(prefix)
                    )  # (vocab_size,)

                    # 屏蔽无效Token（设为-inf）
                    valid_set = set(valid_tokens)
                    masked_logits = torch.full_like(
                        token_logits, float("-inf")
                    )
                    for vt in valid_tokens:
                        vt_int = int(vt)
                        if 0 <= vt_int < len(token_logits):
                            masked_logits[vt_int] = token_logits[vt_int]

                    # 应用温度
                    scaled_logits = masked_logits / self.config.temperature

                    # 计算对数概率
                    log_probs = F.log_softmax(scaled_logits, dim=-1)

                    # 获取top-k候选
                    k = min(beam_width, len(valid_tokens))
                    top_log_probs, top_tokens = torch.topk(log_probs, k)

                    for i in range(k):
                        token = int(top_tokens[i])
                        if token not in valid_set:
                            continue
                        new_prefix = prefix + (token,)
                        new_score = score + float(top_log_probs[i])
                        new_beams.append((new_prefix, new_score))

                if not new_beams:
                    break

                # 保留top-k beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]

                # 提前停止：所有beam都已完成
                if all(self.trie.is_complete_sid(p) for p, _ in beams):
                    break

            # 最终选择：优先选择完整SID，应用长度惩罚
            complete = [(p, s) for p, s in beams if self.trie.is_complete_sid(p)]
            incomplete = [(p, s) for p, s in beams if not self.trie.is_complete_sid(p)]

            if complete:
                # 应用长度惩罚
                scored: List[Tuple[Tuple[int, ...], float]] = []
                for p, s in complete:
                    lp = ((5 + len(p)) / 6) ** self.config.length_penalty
                    scored.append((p, s / lp))
                scored.sort(key=lambda x: x[1], reverse=True)
                final = scored[:num_return]
            else:
                # 未找到完整SID；返回最佳的部分结果
                incomplete.sort(key=lambda x: x[1], reverse=True)
                final = incomplete[:num_return]

            batch_sequences = [p for p, _ in final]
            batch_scores = [s for _, s in final]

            # 如果少于num_return则填充
            while len(batch_sequences) < num_return:
                batch_sequences.append(())
                batch_scores.append(float("-inf"))

            all_batch_sequences.append(batch_sequences)
            all_batch_scores.append(batch_scores)

        return all_batch_sequences, all_batch_scores
