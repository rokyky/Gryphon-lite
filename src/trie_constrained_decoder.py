"""
Trie-constrained beam search decoder for SID generation.

Ensures that the generated SID sequences are valid according to a prefix trie
built from a catalog of known SIDs (Tasks 2.3).

Only explores next tokens that lead to at least one valid complete SID.
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
    """Configuration for trie-constrained beam search.

    Attributes:
        beam_width: number of beams to maintain.
        max_sid_length: max SID token length (default: from model).
        length_penalty: exponent for length normalization (1.0 = neutral).
        temperature: softmax temperature.
    """
    beam_width: int = 10
    max_sid_length: int = 0  # 0 = auto from model
    length_penalty: float = 1.0
    temperature: float = 1.0


class TrieConstrainedBeamSearch:
    """Beam search constrained by a SID trie.

    At each step, only tokens that exist in the trie for the current prefix
    are considered. This guarantees all output SIDs are valid catalog entries.

    Usage:
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
        """Run trie-constrained beam search.

        Args:
            history_sids: (batch, history_len, num_sid_tokens) history sequences.
            model: SID generator model.
            num_return: number of top sequences to return per batch (default: beam_width).

        Returns:
            (sequences, scores):
                sequences: list of lists of SID tuples, one per batch item.
                scores: list of lists of beam scores, one per batch item.
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

            # Initialize beams: (prefix, score)
            beams: List[Tuple[Tuple[int, ...], float]] = [((), 0.0)]

            for step in range(max_len):
                new_beams: List[Tuple[Tuple[int, ...], float]] = []

                for prefix, score in beams:
                    # Get valid next tokens from trie
                    valid_tokens = self.trie.valid_next_tokens(prefix)
                    if not valid_tokens:
                        # This beam cannot continue; keep if it's a complete SID
                        if self.trie.is_complete_sid(prefix):
                            new_beams.append((prefix, score))
                        continue

                    # If this prefix is already a complete SID, keep it as-is
                    if self.trie.is_complete_sid(prefix):
                        new_beams.append((prefix, score))
                        continue

                    # Get model logits for the next token
                    token_logits = model.get_next_token_logits(
                        single_history, list(prefix)
                    )  # (vocab_size,)

                    # Mask out invalid tokens (set to -inf)
                    valid_set = set(valid_tokens)
                    masked_logits = torch.full_like(
                        token_logits, float("-inf")
                    )
                    for vt in valid_tokens:
                        vt_int = int(vt)
                        if 0 <= vt_int < len(token_logits):
                            masked_logits[vt_int] = token_logits[vt_int]

                    # Apply temperature
                    scaled_logits = masked_logits / self.config.temperature

                    # Compute log probabilities
                    log_probs = F.log_softmax(scaled_logits, dim=-1)

                    # Get top-k candidates
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

                # Keep top-k beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]

                # Early stop: all beams are complete
                if all(self.trie.is_complete_sid(p) for p, _ in beams):
                    break

            # Final selection: prefer complete SIDs, apply length penalty
            complete = [(p, s) for p, s in beams if self.trie.is_complete_sid(p)]
            incomplete = [(p, s) for p, s in beams if not self.trie.is_complete_sid(p)]

            if complete:
                # Apply length penalty
                scored: List[Tuple[Tuple[int, ...], float]] = []
                for p, s in complete:
                    lp = ((5 + len(p)) / 6) ** self.config.length_penalty
                    scored.append((p, s / lp))
                scored.sort(key=lambda x: x[1], reverse=True)
                final = scored[:num_return]
            else:
                # No complete SIDs found; return the best partial
                incomplete.sort(key=lambda x: x[1], reverse=True)
                final = incomplete[:num_return]

            batch_sequences = [p for p, _ in final]
            batch_scores = [s for _, s in final]

            # Pad if fewer than num_return
            while len(batch_sequences) < num_return:
                batch_sequences.append(())
                batch_scores.append(float("-inf"))

            all_batch_sequences.append(batch_sequences)
            all_batch_scores.append(batch_scores)

        return all_batch_sequences, all_batch_scores
