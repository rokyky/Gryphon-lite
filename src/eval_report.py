"""
Comprehensive evaluation report (Task 5.3).

Sections:
    - Calibration: compare beam likelihood vs empirical relevance
    - Collision: SID collision statistics and impact
    - Diversity: beam item diversity
    - Long-tail: head vs tail item performance
    - Latency: decoding time, rerank time
"""

import logging
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.eval_metrics import hr_at_k, ndcg_at_k, recall_at_k
from src.sid_quality import (
    collision_group_stats,
    beam_diversity,
    duplicate_rate,
)

logger = logging.getLogger(__name__)


class EvalReport:
    """Build a comprehensive evaluation report.

    Collects metrics across multiple dimensions and outputs as a dict
    (for JSON export) or formatted string.
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}

    # ----- Calibration (Task 5.3) -----

    def compute_calibration(
        self,
        beam_scores: List[float],
        scorer_scores: List[float],
        relevance_labels: List[int],
    ) -> Dict[str, Any]:
        """Compare beam likelihood vs empirical relevance.

        Args:
            beam_scores: beam likelihood scores for each candidate.
            scorer_scores: item-level scorer scores.
            relevance_labels: binary relevance (1 = relevant, 0 = not).

        Returns:
            Dict with calibration metrics.
        """
        if not beam_scores or len(beam_scores) != len(scorer_scores):
            return {}

        beam_order = np.argsort(-np.array(beam_scores))
        scorer_order = np.argsort(-np.array(scorer_scores))

        # Spearman correlation between beam and scorer rankings
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(beam_order, scorer_order)

        # Ranking gap: how many positions does the best item shift
        best_idx = int(np.argmax(relevance_labels)) if max(relevance_labels) > 0 else 0
        beam_rank_of_best = int(np.where(beam_order == best_idx)[0][0]) if best_idx < len(beam_order) else 0
        scorer_rank_of_best = int(np.where(scorer_order == best_idx)[0][0]) if best_idx < len(scorer_order) else 0

        # Score correlation (Pearson)
        from scipy.stats import pearsonr
        pearson_r, p_pearson = pearsonr(beam_scores, scorer_scores)

        return {
            "spearman_correlation": float(corr),
            "spearman_p_value": float(p_value),
            "pearson_correlation": float(pearson_r),
            "pearson_p_value": float(p_pearson),
            "beam_rank_of_best": beam_rank_of_best,
            "scorer_rank_of_best": scorer_rank_of_best,
            "ranking_gap": beam_rank_of_best - scorer_rank_of_best,
        }

    # ----- Collision (Task 5.3) -----

    def compute_collision_stats(
        self,
        sid_to_items: Dict[Tuple[int, ...], List[Any]],
        scorer_can_separate: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """Compute collision statistics and separation impact."""
        stats = collision_group_stats(sid_to_items)

        # How many items are in collision groups?
        total_items = sum(len(items) for items in sid_to_items.values())
        colliding_items = sum(len(items) for items in sid_to_items.values() if len(items) > 1)
        collision_item_rate = colliding_items / max(total_items, 1)

        stats["total_items"] = total_items
        stats["colliding_items"] = colliding_items
        stats["collision_item_rate"] = collision_item_rate

        # Scorer separation impact
        if scorer_can_separate:
            sep_counts = Counter(scorer_can_separate.values())
            stats["scorer_separation"] = {
                "groups_separated": sep_counts.get(True, 0),
                "groups_not_separated": sep_counts.get(False, 0),
                "separation_rate": sep_counts.get(True, 0) / max(len(scorer_can_separate), 1),
            }

        return stats

    # ----- Diversity (Task 5.3) -----

    def compute_diversity(
        self,
        generated_items_by_user: Dict[Any, List[Any]],
    ) -> Dict[str, float]:
        """Compute beam diversity metrics across all users."""
        diversities = []
        for user_id, items in generated_items_by_user.items():
            diversities.append(beam_diversity(items))

        return {
            "avg_beam_diversity": float(np.mean(diversities)) if diversities else 0.0,
            "min_beam_diversity": float(np.min(diversities)) if diversities else 0.0,
            "max_beam_diversity": float(np.max(diversities)) if diversities else 0.0,
        }

    # ----- Long-tail (Task 5.3) -----

    def compute_long_tail(
        self,
        ranked_lists_by_user: Dict[Any, List[Any]],
        ground_truth_by_user: Dict[Any, List[Any]],
        item_popularity: Dict[Any, int],
        tail_threshold: int = 20,
    ) -> Dict[str, Any]:
        """Compute head vs tail item performance.

        Items with popularity <= tail_threshold percentile are "tail".

        Args:
            ranked_lists_by_user: dict of user -> ranked item list.
            ground_truth_by_user: dict of user -> ground truth items.
            item_popularity: dict of item_id -> interaction count.
            tail_threshold: percentile threshold for tail definition.

        Returns:
            Dict with head and tail HR/NDCG.
        """
        if not item_popularity:
            return {}

        pop_values = np.array(list(item_popularity.values()))
        tail_cutoff = np.percentile(pop_values, tail_threshold)

        tail_items = {item for item, pop in item_popularity.items() if pop <= tail_cutoff}
        head_items = {item for item, pop in item_popularity.items() if pop > tail_cutoff}

        head_hits = 0
        head_total = 0
        tail_hits = 0
        tail_total = 0
        head_ndcg_sum = 0.0
        tail_ndcg_sum = 0.0

        for user_id, ranked_list in ranked_lists_by_user.items():
            gt = ground_truth_by_user.get(user_id, [])
            if not gt:
                continue

            target = gt[0]  # single target

            # Head
            if target in head_items:
                head_total += 1
                hit = hr_at_k(ranked_list, [target], k=10)
                head_hits += hit
                if hit:
                    head_ndcg_sum += ndcg_at_k(ranked_list, [target], k=10)

            # Tail
            if target in tail_items:
                tail_total += 1
                hit = hr_at_k(ranked_list, [target], k=10)
                tail_hits += hit
                if hit:
                    tail_ndcg_sum += ndcg_at_k(ranked_list, [target], k=10)

        return {
            "tail_threshold_percentile": tail_threshold,
            "tail_population_cutoff": int(tail_cutoff),
            "num_head_items": len(head_items),
            "num_tail_items": len(tail_items),
            "head": {
                "hr@10": head_hits / max(head_total, 1),
                "ndcg@10": head_ndcg_sum / max(head_total, 1),
                "num_samples": head_total,
            },
            "tail": {
                "hr@10": tail_hits / max(tail_total, 1),
                "ndcg@10": tail_ndcg_sum / max(tail_total, 1),
                "num_samples": tail_total,
            },
        }

    # ----- Latency (Task 5.3) -----

    @staticmethod
    def measure_latency(
        decode_fn,
        rerank_fn=None,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """Measure decoding and reranking latency.

        Args:
            decode_fn: callable that runs decoding and returns results.
            rerank_fn: optional callable that runs reranking.
            num_runs: number of warm-up + measured runs.

        Returns:
            Dict with latency stats in milliseconds.
        """
        latencies = {"decode": [], "rerank": []}

        for _ in range(num_runs):
            # Decoding
            start = time.perf_counter()
            decode_fn()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies["decode"].append(elapsed)

            # Reranking
            if rerank_fn:
                start = time.perf_counter()
                rerank_fn()
                elapsed = (time.perf_counter() - start) * 1000
                latencies["rerank"].append(elapsed)

        result = {}
        for key, values in latencies.items():
            if values:
                result[f"{key}_mean_ms"] = float(np.mean(values))
                result[f"{key}_median_ms"] = float(np.median(values))
                result[f"{key}_std_ms"] = float(np.std(values))
                result[f"{key}_min_ms"] = float(np.min(values))
                result[f"{key}_max_ms"] = float(np.max(values))

        return result

    # ----- Full report -----

    def generate_report(
        self,
        calibration: Optional[Dict[str, Any]] = None,
        collision: Optional[Dict[str, Any]] = None,
        diversity: Optional[Dict[str, Any]] = None,
        long_tail: Optional[Dict[str, Any]] = None,
        latency: Optional[Dict[str, Any]] = None,
        ranking_comparison: Optional[Dict[str, Any]] = None,
        generation_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Combine all sections into a single report."""
        report = {}

        if calibration:
            report["calibration"] = calibration
        if collision:
            report["collision"] = collision
        if diversity:
            report["diversity"] = diversity
        if long_tail:
            report["long_tail"] = long_tail
        if latency:
            report["latency"] = latency
        if ranking_comparison:
            report["ranking_comparison"] = ranking_comparison
        if generation_metrics:
            report["generation_metrics"] = generation_metrics

        self.results = report
        return report

# ===== Standalone Report Classes (Task 5.3) =====


class CalibrationReport:
    """Calibration report: compare beam likelihood vs item-level scorer rankings.

    Measures how well beam likelihood scores correlate with item relevance.
    """

    def __init__(self):
        self.report = EvalReport()

    def compute(self, beam_scores, scorer_scores, relevance_labels):
        """Compute calibration metrics.

        Args:
            beam_scores: list of beam likelihood scores.
            scorer_scores: list of scorer relevance scores.
            relevance_labels: binary relevance (1=relevant, 0=not).

        Returns:
            Dict with calibration metrics.
        """
        return self.report.compute_calibration(beam_scores, scorer_scores, relevance_labels)

    def to_string(self, metrics):
        """Format calibration metrics as a string."""
        lines = ["=== Calibration Report ==="]
        if not metrics:
            lines.append("  No calibration data.")
            return "\n".join(lines)
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class CollisionReport:
    """Collision report: SID collision statistics and scorer separation impact."""

    def __init__(self):
        self.report = EvalReport()

    def compute(self, sid_to_items, scorer_can_separate=None):
        """Compute collision statistics.

        Args:
            sid_to_items: mapping from SID tuple to list of item IDs.
            scorer_can_separate: optional dict of SID -> whether scorer separates items.

        Returns:
            Dict with collision metrics.
        """
        return self.report.compute_collision_stats(sid_to_items, scorer_can_separate)

    def to_string(self, metrics):
        """Format collision metrics as a string."""
        lines = ["=== Collision Report ==="]
        if not metrics:
            lines.append("  No collision data.")
            return "\n".join(lines)
        for key, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        lines.append(f"    {k}: {v:.4f}")
                    else:
                        lines.append(f"    {k}: {v}")
            elif isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class DiversityReport:
    """Diversity report: beam diversity statistics across users."""

    def __init__(self):
        self.report = EvalReport()

    def compute(self, generated_items_by_user):
        """Compute diversity metrics.

        Args:
            generated_items_by_user: dict of user -> list of generated item IDs.

        Returns:
            Dict with diversity metrics.
        """
        return self.report.compute_diversity(generated_items_by_user)

    def to_string(self, metrics):
        """Format diversity metrics as a string."""
        lines = ["=== Diversity Report ==="]
        if not metrics:
            lines.append("  No diversity data.")
            return "\n".join(lines)
        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)


class LatencyReport:
    """Latency report: decoding and reranking timing measurements."""

    @staticmethod
    def measure(decode_fn, rerank_fn=None, num_runs=10):
        """Measure latency.

        Args:
            decode_fn: callable for decoding.
            rerank_fn: optional callable for reranking.
            num_runs: number of measurement runs.

        Returns:
            Dict with latency metrics in milliseconds.
        """
        return EvalReport.measure_latency(decode_fn, rerank_fn, num_runs)

    @staticmethod
    def to_string(metrics):
        """Format latency metrics as a string."""
        lines = ["=== Latency Report ==="]
        if not metrics:
            lines.append("  No latency data.")
            return "\n".join(lines)
        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.2f}")
        return "\n".join(lines)


def full_report(
    calibration_metrics=None,
    collision_metrics=None,
    diversity_metrics=None,
    long_tail_metrics=None,
    latency_metrics=None,
    ranking_comparison=None,
    generation_metrics=None,
    output_path=None,
):
    """Generate a full evaluation report combining all sections.

    Args:
        calibration_metrics: dict from CalibrationReport.compute().
        collision_metrics: dict from CollisionReport.compute().
        diversity_metrics: dict from DiversityReport.compute().
        long_tail_metrics: dict from EvalReport.compute_long_tail().
        latency_metrics: dict from LatencyReport.measure().
        ranking_comparison: dict from ranking comparison.
        generation_metrics: dict from generation quality evaluation.
        output_path: optional path to save JSON report.

    Returns:
        Tuple of (report_dict, formatted_string).
    """
    report = EvalReport()
    report.generate_report(
        calibration=calibration_metrics,
        collision=collision_metrics,
        diversity=diversity_metrics,
        long_tail=long_tail_metrics,
        latency=latency_metrics,
        ranking_comparison=ranking_comparison,
        generation_metrics=generation_metrics,
    )

    # Build formatted string
    lines = []
    lines.append("=" * 60)
    lines.append("Gryphon-lite Full Evaluation Report")
    lines.append("=" * 60)

    sections = {
        "Calibration": calibration_metrics,
        "Collision": collision_metrics,
        "Diversity": diversity_metrics,
        "Long Tail": long_tail_metrics,
        "Latency": latency_metrics,
        "Ranking Comparison": ranking_comparison,
        "Generation Metrics": generation_metrics,
    }

    for section_name, metrics in sections.items():
        if not metrics:
            continue
        lines.append(f"\n--- {section_name.upper()} ---")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, dict):
                    lines.append(f"  {key}:")
                    for k, v in value.items():
                        if isinstance(v, float):
                            lines.append(f"    {k}: {v:.4f}")
                        else:
                            lines.append(f"    {k}: {v}")
                elif isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("=" * 60)
    formatted = "\n".join(lines)

    # Save to JSON if path provided
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.results, f, indent=2)

    return report.results, formatted
