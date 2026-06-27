"""
全面的评估报告（任务5.3）。

章节：
    - Calibration（校准）：比较beam似然度与经验相关性
    - Collision（冲突）：SID冲突统计及影响
    - Diversity（多样性）：Beam物品多样性
    - Long-tail（长尾）：头部与尾部物品的表现
    - Latency（延迟）：解码时间、重排序时间
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
    """构建全面的评估报告。

    在多个维度上收集指标，并以字典形式（用于JSON导出）或格式化字符串输出。
    """

    def __init__(self):
        self.results: Dict[str, Any] = {}

    # ----- 校准（任务5.3） -----

    def compute_calibration(
        self,
        beam_scores: List[float],
        scorer_scores: List[float],
        relevance_labels: List[int],
    ) -> Dict[str, Any]:
        """比较beam似然度与经验相关性。

        Args:
            beam_scores: 每个候选的beam似然分数。
            scorer_scores: 物品级评分器的分数。
            relevance_labels: 二元相关性（1=相关，0=不相关）。

        Returns:
            包含校准指标的字典。
        """
        if not beam_scores or len(beam_scores) != len(scorer_scores):
            return {}

        beam_order = np.argsort(-np.array(beam_scores))
        scorer_order = np.argsort(-np.array(scorer_scores))

        # Spearman相关性
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(beam_order, scorer_order)

        # 排名差距：最佳物品移动了多少个位置
        best_idx = int(np.argmax(relevance_labels)) if max(relevance_labels) > 0 else 0
        beam_rank_of_best = int(np.where(beam_order == best_idx)[0][0]) if best_idx < len(beam_order) else 0
        scorer_rank_of_best = int(np.where(scorer_order == best_idx)[0][0]) if best_idx < len(scorer_order) else 0

        # 分数相关性（Pearson）
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

    # ----- 冲突（任务5.3） -----

    def compute_collision_stats(
        self,
        sid_to_items: Dict[Tuple[int, ...], List[Any]],
        scorer_can_separate: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """计算冲突统计信息及分离影响。"""
        stats = collision_group_stats(sid_to_items)

        # 有多少物品在冲突组中？
        total_items = sum(len(items) for items in sid_to_items.values())
        colliding_items = sum(len(items) for items in sid_to_items.values() if len(items) > 1)
        collision_item_rate = colliding_items / max(total_items, 1)

        stats["total_items"] = total_items
        stats["colliding_items"] = colliding_items
        stats["collision_item_rate"] = collision_item_rate

        # 评分器分离影响
        if scorer_can_separate:
            sep_counts = Counter(scorer_can_separate.values())
            stats["scorer_separation"] = {
                "groups_separated": sep_counts.get(True, 0),
                "groups_not_separated": sep_counts.get(False, 0),
                "separation_rate": sep_counts.get(True, 0) / max(len(scorer_can_separate), 1),
            }

        return stats

    # ----- 多样性（任务5.3） -----

    def compute_diversity(
        self,
        generated_items_by_user: Dict[Any, List[Any]],
    ) -> Dict[str, float]:
        """计算所有用户的Beam多样性指标。"""
        diversities = []
        for user_id, items in generated_items_by_user.items():
            diversities.append(beam_diversity(items))

        return {
            "avg_beam_diversity": float(np.mean(diversities)) if diversities else 0.0,
            "min_beam_diversity": float(np.min(diversities)) if diversities else 0.0,
            "max_beam_diversity": float(np.max(diversities)) if diversities else 0.0,
        }

    # ----- 长尾（任务5.3） -----

    def compute_long_tail(
        self,
        ranked_lists_by_user: Dict[Any, List[Any]],
        ground_truth_by_user: Dict[Any, List[Any]],
        item_popularity: Dict[Any, int],
        tail_threshold: int = 20,
    ) -> Dict[str, Any]:
        """计算头部与尾部物品的表现。

        流行度 <= tail_threshold百分位的物品被视为"尾部"。

        Args:
            ranked_lists_by_user: 用户到排序物品列表的字典。
            ground_truth_by_user: 用户到真实物品的字典。
            item_popularity: 物品ID到交互次数的字典。
            tail_threshold: 尾部定义的百分位阈值。

        Returns:
            包含头部和尾部HR/NDCG的字典。
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

            target = gt[0]  # 单个目标

            # 头部
            if target in head_items:
                head_total += 1
                hit = hr_at_k(ranked_list, [target], k=10)
                head_hits += hit
                if hit:
                    head_ndcg_sum += ndcg_at_k(ranked_list, [target], k=10)

            # 尾部
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

    # ----- 延迟（任务5.3） -----

    @staticmethod
    def measure_latency(
        decode_fn,
        rerank_fn=None,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """测量解码和重排序延迟。

        Args:
            decode_fn: 运行解码并返回结果的可调用对象。
            rerank_fn: 可选，运行重排序的可调用对象。
            num_runs: 预热+测量运行的次数。

        Returns:
            以毫秒为单位的延迟统计字典。
        """
        latencies = {"decode": [], "rerank": []}

        for _ in range(num_runs):
            # 解码
            start = time.perf_counter()
            decode_fn()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies["decode"].append(elapsed)

            # 重排序
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
        """将所有章节合并为单个报告。"""
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

# ===== 独立的报告类（任务5.3） =====


class CalibrationReport:
    """校准报告：比较beam似然度与物品级评分器排名。

    衡量beam似然分数与物品相关性的相关程度。
    """

    def __init__(self):
        self.report = EvalReport()

    def compute(self, beam_scores, scorer_scores, relevance_labels):
        """计算校准指标。

        Args:
            beam_scores: beam似然分数列表。
            scorer_scores: 评分器相关性分数列表。
            relevance_labels: 二元相关性（1=相关，0=不相关）。

        Returns:
            包含校准指标的字典。
        """
        return self.report.compute_calibration(beam_scores, scorer_scores, relevance_labels)

    def to_string(self, metrics):
        """将校准指标格式化为字符串。"""
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
    """冲突报告：SID冲突统计信息和评分器分离影响。"""

    def __init__(self):
        self.report = EvalReport()

    def compute(self, sid_to_items, scorer_can_separate=None):
        """计算冲突统计信息。

        Args:
            sid_to_items: 从SID元组到物品ID列表的映射。
            scorer_can_separate: 可选的SID到评分器是否能区分物品的字典。

        Returns:
            包含冲突指标的字典。
        """
        return self.report.compute_collision_stats(sid_to_items, scorer_can_separate)

    def to_string(self, metrics):
        """将冲突指标格式化为字符串。"""
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
    """多样性报告：所有用户的Beam多样性统计。"""

    def __init__(self):
        self.report = EvalReport()

    def compute(self, generated_items_by_user):
        """计算多样性指标。

        Args:
            generated_items_by_user: 用户到生成物品ID列表的字典。

        Returns:
            包含多样性指标的字典。
        """
        return self.report.compute_diversity(generated_items_by_user)

    def to_string(self, metrics):
        """将多样性指标格式化为字符串。"""
        lines = ["=== Diversity Report ==="]
        if not metrics:
            lines.append("  No diversity data.")
            return "\n".join(lines)
        for key, value in metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)


class LatencyReport:
    """延迟报告：解码和重排序的时间测量。"""

    @staticmethod
    def measure(decode_fn, rerank_fn=None, num_runs=10):
        """测量延迟。

        Args:
            decode_fn: 用于解码的可调用对象。
            rerank_fn: 可选，用于重排序的可调用对象。
            num_runs: 测量运行次数。

        Returns:
            以毫秒为单位的延迟指标字典。
        """
        return EvalReport.measure_latency(decode_fn, rerank_fn, num_runs)

    @staticmethod
    def to_string(metrics):
        """将延迟指标格式化为字符串。"""
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
    """生成结合所有章节的完整评估报告。

    Args:
        calibration_metrics: 来自CalibrationReport.compute()的字典。
        collision_metrics: 来自CollisionReport.compute()的字典。
        diversity_metrics: 来自DiversityReport.compute()的字典。
        long_tail_metrics: 来自EvalReport.compute_long_tail()的字典。
        latency_metrics: 来自LatencyReport.measure()的字典。
        ranking_comparison: 来自排名比较的字典。
        generation_metrics: 来自生成质量评估的字典。
        output_path: 可选的JSON报告保存路径。

    Returns:
        (report_dict, formatted_string)元组。
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

    # 构建格式化字符串
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
