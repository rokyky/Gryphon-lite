#!/usr/bin/env python3
"""
比较排序方法：beam似然度 vs 物品级评分器（任务3.5）。

报告：
    - HR/NDCG差异
    - 冲突物品分离（评分器能否区分同一SID组中的物品？）
    - 排名差距指标

用法：
    python scripts/compare_ranking.py \\
        --test_path data/test.csv \\
        --index_path data/indices.json \\
        --sid_generator_ckpt checkpoints/sid_generator/best_model.pt \\
        --scorer_ckpt checkpoints/item_scorer/best_model.pt \\
        --output results/ranking_comparison.json
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# 将项目根目录添加到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval_metrics import hr_at_k, ndcg_at_k
from src.item_scorer import ItemScorerConfig, create_item_scorer
from src.sid_generator import SIDGenerator, SIDGeneratorConfig
from src.sid_mapper import SIDTrie, build_sid_trie
from src.trie_constrained_decoder import TrieConstrainedBeamSearch, TrieBeamSearchConfig
from src.sid_quality import compute_generation_metrics

logger = logging.getLogger(__name__)


def load_test_data(test_path, index_path, max_history_len=50):
    """加载测试序列并返回结构化样本。"""
    import pandas as pd
    with open(index_path, "r") as f:
        index = json.load(f)
    item_to_sid = {iid: tuple(int(s) for s in sid_list) for iid, sid_list in index.items()}
    sid_to_items = defaultdict(list)
    for iid, sid in item_to_sid.items():
        sid_to_items[sid].append(iid)

    data = pd.read_csv(test_path)
    samples = []
    for idx in range(len(data)):
        row = data.iloc[idx]
        try:
            history_ids = eval(str(row.get("history_item_id", "[]")))
        except:
            history_ids = []
        target_id = str(row.get("item_id", ""))

        history_sids = [item_to_sid.get(str(h)) for h in history_ids if str(h) in item_to_sid]
        target_sid = item_to_sid.get(target_id)
        if len(history_sids) < 1 or target_sid is None:
            continue
        if len(history_sids) > max_history_len:
            history_sids = history_sids[-max_history_len:]

        samples.append({
            "history_sids": history_sids,
            "target_item_id": target_id,
            "target_sid": target_sid,
        })

    logger.info(f"Loaded {len(samples)} test samples")
    return samples, item_to_sid, sid_to_items


@torch.no_grad()
def beam_ranking(
    history_sids,
    model,
    decoder,
    sid_to_items,
    beam_width=20,
):
    """仅根据beam似然度对候选项排序。"""
    B = 1
    H = len(history_sids)
    T = model.config.num_sid_tokens

    hist_tensor = torch.tensor([[history_sids]], dtype=torch.long)
    if torch.cuda.is_available():
        hist_tensor = hist_tensor.cuda()

    # 需要重塑：(1, H, T)
    hist = hist_tensor.reshape(B, H, T)

    sequences, scores = decoder.search(hist, model, num_return=beam_width)

    # 将SID映射到物品
    ranked_items = []
    for sid, score in zip(sequences[0], scores[0]):
        if sid in sid_to_items:
            for item_id in sid_to_items[sid]:
                ranked_items.append((item_id, score))

    return ranked_items


@torch.no_grad()
def scorer_ranking(
    history_sids,
    target_sid,
    model,
    scorer,
    sid_to_items,
    item_to_sid,
    decoder,
    beam_width=20,
):
    """根据物品级评分器对候选项排序。"""
    # 首先获取beam候选
    B = 1
    H = len(history_sids)
    T = model.config.num_sid_tokens

    hist_tensor = torch.tensor([[history_sids]], dtype=torch.long)
    if torch.cuda.is_available():
        hist_tensor = hist_tensor.cuda()
    hist = hist_tensor.reshape(B, H, T)

    sequences, beam_scores = decoder.search(hist, model, num_return=beam_width)
    device = hist.device

    # 获取用户嵌入
    user_emb = model.token_embedding(hist.reshape(1, -1)).mean(dim=1)

    # 对每个候选项评分
    scored_items = []
    for sid, bs in zip(sequences[0], beam_scores):
        if sid not in sid_to_items:
            continue
        for item_id in sid_to_items[sid]:
            # 获取物品和SID嵌入
            sid_tensor = torch.tensor([sid], dtype=torch.long, device=device)
            sid_emb = model.token_embedding(sid_tensor).mean(dim=1).unsqueeze(0)

            item_id_hash = hash(str(item_id)) % 100000
            item_id_tensor = torch.tensor([[item_id_hash]], device=device)

            scorer_score = scorer(
                user_embeddings=user_emb.unsqueeze(0),
                item_ids=item_id_tensor,
                sid_embeddings=sid_emb.unsqueeze(0),
            )
            scored_items.append((item_id, float(scorer_score[0, 0])))

    scored_items.sort(key=lambda x: x[1], reverse=True)
    return scored_items


def compute_ranking_gap(beam_items, scorer_items, target_item_id):
    """计算比较beam和评分器排名的指标。"""
    result = {}

    # 目标物品在每种排序中的位置
    beam_positions = [i for i, (item_id, _) in enumerate(beam_items) if item_id == target_item_id]
    scorer_positions = [i for i, (item_id, _) in enumerate(scorer_items) if item_id == target_item_id]

    result["beam_position"] = beam_positions[0] if beam_positions else -1
    result["scorer_position"] = scorer_positions[0] if scorer_positions else -1
    result["position_change"] = (
        result["beam_position"] - result["scorer_position"]
        if beam_positions and scorer_positions else None
    )

    # HR/NDCG（假设beam_width是候选池大小）
    K = min(10, len(beam_items), len(scorer_items))

    beam_candidate_ids = [item_id for item_id, _ in beam_items[:K]]
    scorer_candidate_ids = [item_id for item_id, _ in scorer_items[:K]]

    result[f"beam_hr@{K}"] = hr_at_k(beam_candidate_ids, [target_item_id])
    result[f"scorer_hr@{K}"] = hr_at_k(scorer_candidate_ids, [target_item_id])
    result[f"beam_ndcg@{K}"] = ndcg_at_k(beam_candidate_ids, [target_item_id])
    result[f"scorer_ndcg@{K}"] = ndcg_at_k(scorer_candidate_ids, [target_item_id])

    return result


def compute_collision_separation(beam_items, scorer_items, sid_to_items):
    """检查评分器能否区分共享同一SID的物品。"""
    beam_sids_used = defaultdict(list)
    for item_id, score in beam_items:
        for sid, items in sid_to_items.items():
            if item_id in items:
                beam_sids_used[sid].append((item_id, score))
                break

    scorer_sids_used = defaultdict(list)
    for item_id, score in scorer_items:
        for sid, items in sid_to_items.items():
            if item_id in items:
                scorer_sids_used[sid].append((item_id, score))
                break

    # 统计beam中任意排序但评分器重新排序的冲突数
    beam_arbitrary = 0
    scorer_different = 0
    for sid in beam_sids_used:
        if len(beam_sids_used[sid]) > 1:
            beam_arbitrary += 1
            beam_order = [item_id for item_id, _ in beam_sids_used[sid]]
            scorer_order = [item_id for item_id, _ in scorer_sids_used.get(sid, [])]
            if beam_order != scorer_order:
                scorer_different += 1

    return {
        "collision_groups_with_beam": beam_arbitrary,
        "collision_groups_reranked_by_scorer": scorer_different,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare beam likelihood vs item scorer ranking")
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--sid_generator_ckpt", type=str, required=True)
    parser.add_argument("--scorer_ckpt", type=str, default=None)
    parser.add_argument("--beam_width", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=500,
                        help="要评估的测试样本数量")
    parser.add_argument("--output", type=str, default="results/ranking_comparison.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 加载数据
    samples, item_to_sid, sid_to_items = load_test_data(args.test_path, args.index_path)
    if args.num_samples and args.num_samples < len(samples):
        samples = random.sample(samples, args.num_samples)
    logger.info(f"Evaluating on {len(samples)} samples")

    # 构建Trie
    trie = build_sid_trie(dict(sid_to_items))

    # 加载SID生成器
    ckpt = torch.load(args.sid_generator_ckpt, map_location="cpu")
    cfg = ckpt.get("config", SIDGeneratorConfig())
    model = SIDGenerator(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 设置解码器
    decoder_config = TrieBeamSearchConfig(beam_width=args.beam_width)
    decoder = TrieConstrainedBeamSearch(trie, decoder_config)

    # 加载评分器（可选；如果未提供，仅使用beam排序）
    scorer = None
    if args.scorer_ckpt:
        scorer_ckpt = torch.load(args.scorer_ckpt, map_location="cpu")
        scorer_cfg = scorer_ckpt.get("config", ItemScorerConfig())
        scorer = create_item_scorer(scorer_cfg, num_items=100000).to(device)
        scorer.load_state_dict(scorer_ckpt["model_state_dict"])
        scorer.eval()

    # 评估
    beam_hr_list = []
    scorer_hr_list = []
    beam_ndcg_list = []
    scorer_ndcg_list = []
    position_changes = []
    collision_sep_total = {"collision_groups_with_beam": 0, "collision_groups_reranked_by_scorer": 0}
    gen_metrics_list = []

    for sample in tqdm(samples, desc="Comparing rankings"):
        history_sids = sample["history_sids"]
        target_item_id = sample["target_item_id"]
        target_sid = sample["target_sid"]

        # Beam排序
        beam_items = beam_ranking(history_sids, model, decoder, sid_to_items, args.beam_width)

        # 生成指标
        gen_sids = [sid for sid, _ in zip(
            decoder.search(
                torch.tensor([[[history_sids]]], dtype=torch.long).reshape(1, len(history_sids), model.config.num_sid_tokens).to(device) if torch.cuda.is_available() else torch.tensor([[[history_sids]]], dtype=torch.long).reshape(1, len(history_sids), model.config.num_sid_tokens),
                model, num_return=args.beam_width
            )[0][0],
            [0] * args.beam_width
        )]
        gen_item_ids = [item_id for item_id, _ in beam_items]
        gen_metrics = compute_generation_metrics(
            gen_sids, gen_item_ids, trie, sid_to_items
        )
        gen_metrics_list.append(gen_metrics)

        # 计算仅beam的指标
        beam_candidate_ids = [item_id for item_id, _ in beam_items[:10]]
        beam_hr_list.append(hr_at_k(beam_candidate_ids, [target_item_id]))
        beam_ndcg_list.append(ndcg_at_k(beam_candidate_ids, [target_item_id]))

        # 评分器排序
        if scorer is not None:
            scorer_items = scorer_ranking(
                history_sids, target_sid, model, scorer, sid_to_items, item_to_sid, decoder, args.beam_width
            )

            scorer_candidate_ids = [item_id for item_id, _ in scorer_items[:10]]
            scorer_hr_list.append(hr_at_k(scorer_candidate_ids, [target_item_id]))
            scorer_ndcg_list.append(ndcg_at_k(scorer_candidate_ids, [target_item_id]))

            # 位置变化
            beam_pos = next((i for i, (iid, _) in enumerate(beam_items) if iid == target_item_id), -1)
            scorer_pos = next((i for i, (iid, _) in enumerate(scorer_items) if iid == target_item_id), -1)
            if beam_pos >= 0 and scorer_pos >= 0:
                position_changes.append(beam_pos - scorer_pos)

            # 冲突分离
            coll_sep = compute_collision_separation(beam_items, scorer_items, sid_to_items)
            for k in coll_sep:
                collision_sep_total[k] += coll_sep[k]
        else:
            scorer_hr_list.append(0.0)
            scorer_ndcg_list.append(0.0)

    # 汇总结果
    results = {
        "num_samples": len(samples),
        "beam_width": args.beam_width,
        "beam": {
            "avg_hr@10": float(np.mean(beam_hr_list)) if beam_hr_list else 0.0,
            "avg_ndcg@10": float(np.mean(beam_ndcg_list)) if beam_ndcg_list else 0.0,
        },
        "scorer": {
            "avg_hr@10": float(np.mean(scorer_hr_list)) if scorer_hr_list else 0.0,
            "avg_ndcg@10": float(np.mean(scorer_ndcg_list)) if scorer_ndcg_list else 0.0,
        },
        "ranking_gap": {
            "hr_improvement": (
                float(np.mean(scorer_hr_list) - np.mean(beam_hr_list))
                if scorer_hr_list else None
            ),
            "ndcg_improvement": (
                float(np.mean(scorer_ndcg_list) - np.mean(beam_ndcg_list))
                if scorer_ndcg_list else None
            ),
            "avg_position_change": float(np.mean(position_changes)) if position_changes else None,
            "position_improved": sum(1 for c in position_changes if c > 0) if position_changes else 0,
            "position_worsened": sum(1 for c in position_changes if c < 0) if position_changes else 0,
        },
        "collision_separation": collision_sep_total,
        "generation_metrics": {
            k: float(np.mean([g[k] for g in gen_metrics_list]))
            for k in gen_metrics_list[0]
        } if gen_metrics_list else {},
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("Ranking Comparison Results")
    logger.info("=" * 60)
    logger.info(f"Beam HR@10: {results['beam']['avg_hr@10']:.4f}")
    logger.info(f"Beam NDCG@10: {results['beam']['avg_ndcg@10']:.4f}")
    if scorer is not None:
        logger.info(f"Scorer HR@10: {results['scorer']['avg_hr@10']:.4f}")
        logger.info(f"Scorer NDCG@10: {results['scorer']['avg_ndcg@10']:.4f}")
        logger.info(f"HR Improvement: {results['ranking_gap']['hr_improvement']:+.4f}")
        logger.info(f"NDCG Improvement: {results['ranking_gap']['ndcg_improvement']:+.4f}")
    logger.info(f"Collision groups with beam: {collision_sep_total['collision_groups_with_beam']}")
    logger.info(f"Collision groups reranked: {collision_sep_total['collision_groups_reranked_by_scorer']}")
    logger.info(f"Generation: valid_sid={results['generation_metrics'].get('valid_sid_rate', 0):.4f}")
    logger.info("=" * 60)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
