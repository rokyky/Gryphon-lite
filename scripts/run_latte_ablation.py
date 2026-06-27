#!/usr/bin/env python3
"""
Latte消融实验：比较vanilla SID生成器与潜在Token SID生成器（任务4.3）。

测量：
    - HR@10 / HR@20
    - NDCG@10 / NDCG@20
    - 冲突分离
    - 有效物品率
    - 解码延迟

打印消融比较表。

用法：
    python scripts/run_latte_ablation.py \
        --test_path data/test.csv \
        --index_path data/indices.json \
        --sid_generator_ckpt checkpoints/sid_generator/best_model.pt \
        --output results/latte_ablation.json
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# 将项目根目录添加到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval_metrics import hr_at_k, ndcg_at_k
from src.sid_generator import SIDGenerator, SIDGeneratorConfig
from src.sid_mapper import SIDTrie, build_sid_trie
from src.trie_constrained_decoder import TrieConstrainedBeamSearch, TrieBeamSearchConfig
from src.sid_quality import valid_sid_rate, valid_item_rate, beam_diversity

logger = logging.getLogger(__name__)


def load_test_data(test_path, index_path, max_history_len=50, min_seq_len=1):
    """加载带有SID映射的测试序列。"""
    import pandas as pd
    with open(index_path, 'r') as f:
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
            history_ids = eval(str(row.get('history_item_id', '[]')))
        except (ValueError, SyntaxError):
            history_ids = []
        target_id = str(row.get('item_id', ''))

        history_sids = [item_to_sid.get(str(h)) for h in history_ids if str(h) in item_to_sid]
        history_sids = [s for s in history_sids if s is not None]
        target_sid = item_to_sid.get(str(target_id))

        if len(history_sids) < min_seq_len or target_sid is None:
            continue
        if len(history_sids) > max_history_len:
            history_sids = history_sids[-max_history_len:]

        samples.append({
            'history_sids': history_sids,
            'target_item_id': str(target_id),
            'target_sid': target_sid,
        })

    logger.info(f'Loaded {len(samples)} test samples')
    return samples, item_to_sid, dict(sid_to_items)


def build_model(variant, ckpt_path, device, num_sid_tokens=3, vocab_per_token=256,
                hidden_dim=128, num_layers=3, num_heads=4, latent_token_count=0):
    """构建SID生成器模型变体。

    Args:
        variant: 'vanilla'或'latent'
        ckpt_path: 检查点路径（公平比较共享权重）。
        latent_token_count: 潜在Token数量（vanilla为0，latent为4）。
    """
    use_latent = variant == 'latent' and latent_token_count > 0
    ltc = latent_token_count if use_latent else 0

    config = SIDGeneratorConfig(
        vocab_size_per_token=vocab_per_token,
        num_sid_tokens=num_sid_tokens,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_latent_tokens=use_latent,
        latent_token_count=ltc,
        latent_token_dim=hidden_dim,
    )

    model = SIDGenerator(config).to(device)

    # 加载检查点权重（变体之间共享）
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['model_state_dict']
        # 过滤到匹配的键（如果加载vanilla权重则忽略潜在Token）
        model_dict = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict
                    and model_dict[k].shape == v.shape}
        missing = model_dict.keys() - filtered.keys()
        if missing:
            logger.warning(f'{variant}: {len(missing)} keys not in checkpoint '
                          f'(expected if latent weights are new): {list(missing)[:3]}...')
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        logger.info(f'{variant}: loaded checkpoint, {len(filtered)}/{len(model_dict)} keys matched')
    else:
        logger.warning(f'{variant}: checkpoint not found, using random init')

    model.eval()
    return model


@torch.no_grad()
def evaluate_variant(model, test_samples, sid_to_items, trie, beam_width,
                     num_samples, device, measure_latency=True):
    """评估测试样本上的模型变体。

    返回：
        指标的字典。
    """
    beam_cfg = TrieBeamSearchConfig(beam_width=beam_width)
    decoder = TrieConstrainedBeamSearch(trie, beam_cfg)

    hr10_list = []
    hr20_list = []
    ndcg10_list = []
    ndcg20_list = []
    generated_sids_list = []
    generated_item_ids_list = []
    latencies = []
    collision_sep = {'groups_with_beam': 0, 'groups_separated': 0}

    eval_samples = test_samples[:num_samples] if num_samples else test_samples
    num_sid_tokens = model.config.num_sid_tokens

    for sample in tqdm(eval_samples, desc=f'Evaluating'):
        history_sids = sample['history_sids']
        target_item_id = sample['target_item_id']

        H = len(history_sids)
        hist_tensor = torch.tensor([[history_sids]], dtype=torch.long, device=device)
        hist = hist_tensor.reshape(1, H, num_sid_tokens)

        # 测量延迟
        start = time.perf_counter()
        sequences, scores = decoder.search(hist, model, num_return=beam_width)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

        # 将SID映射到物品
        beam_items = []
        seen = set()
        for sid in sequences[0]:
            if sid in sid_to_items:
                for item_id in sid_to_items[sid]:
                    if item_id not in seen:
                        seen.add(item_id)
                        beam_items.append(item_id)

        generated_sids_list.extend(sequences[0])
        generated_item_ids_list.extend(beam_items)

        # 计算HR/NDCG
        hr10_list.append(hr_at_k(beam_items, [target_item_id], k=10))
        hr20_list.append(hr_at_k(beam_items, [target_item_id], k=20))
        ndcg10_list.append(ndcg_at_k(beam_items, [target_item_id], k=10))
        ndcg20_list.append(ndcg_at_k(beam_items, [target_item_id], k=20))

        # 冲突分离：检查评分器是否会重排冲突组
        # 对于此消融实验，我们简单地统计输出中的SID冲突组
        sid_counts = defaultdict(list)
        for sid in sequences[0]:
            if sid in sid_to_items:
                for item_id in sid_to_items[sid]:
                    sid_counts[sid].append(item_id)
        for sid, items in sid_counts.items():
            if len(items) > 1:
                collision_sep['groups_with_beam'] += 1

    metrics = {
        'HR@10': float(np.mean(hr10_list)) if hr10_list else 0.0,
        'HR@20': float(np.mean(hr20_list)) if hr20_list else 0.0,
        'NDCG@10': float(np.mean(ndcg10_list)) if ndcg10_list else 0.0,
        'NDCG@20': float(np.mean(ndcg20_list)) if ndcg20_list else 0.0,
        'valid_sid_rate': valid_sid_rate(generated_sids_list, trie),
        'valid_item_rate': valid_item_rate(generated_sids_list, sid_to_items),
        'beam_diversity': beam_diversity(generated_item_ids_list),
        'avg_latency_ms': float(np.mean(latencies)) if latencies else 0.0,
        'median_latency_ms': float(np.median(latencies)) if latencies else 0.0,
        'collision_groups': collision_sep['groups_with_beam'],
        'num_samples': len(eval_samples),
    }
    return metrics


def print_ablation_table(vanilla_metrics, latent_metrics):
    """打印格式化的消融比较表。"""
    print('\n' + '=' * 80)
    print('Latte Ablation: Vanilla vs Latent Token SID Generator')
    print('=' * 80)

    # Collect all metric keys
    all_keys = set(vanilla_metrics.keys()) | set(latent_metrics.keys())
    sorted_keys = sorted(all_keys)

    header = f"{'Metric':<30}{'Vanilla':<20}{'Latent':<20}{'Delta':<15}"
    print(header)
    print('-' * 80)

    for key in sorted_keys:
        if key in ('num_samples',):
            continue
        v_val = vanilla_metrics.get(key, 'N/A')
        l_val = latent_metrics.get(key, 'N/A')

        if isinstance(v_val, (int, float)) and isinstance(l_val, (int, float)):
            delta = l_val - v_val
            if key in ('avg_latency_ms', 'median_latency_ms'):
                # 对于延迟，负的delta表示改进（越小越好）
                delta_str = f'{delta:+.4f}'
            else:
                delta_str = f'{delta:+.4f}'
            row = f"{key:<30}{v_val:<20.4f}{l_val:<20.4f}{delta_str:<15}"
        else:
            row = f"{key:<30}{str(v_val):<20}{str(l_val):<20}{'':<15}"
        print(row)

    print('=' * 80)

    # Summary
    print('\nSummary:')
    for key in ('HR@10', 'HR@20', 'NDCG@10', 'NDCG@20'):
        v = vanilla_metrics.get(key, 0)
        l = latent_metrics.get(key, 0)
        delta = l - v
        direction = 'IMPROVED' if delta > 0 else 'DEGRADED' if delta < 0 else 'UNCHANGED'
        print(f"  {key}: {direction} ({delta:+.4f})")

    v_lat = vanilla_metrics.get('avg_latency_ms', 0)
    l_lat = latent_metrics.get('avg_latency_ms', 0)
    lat_delta = l_lat - v_lat
    print(f"  Avg Latency: {'INCREASED' if lat_delta > 0 else 'DECREASED' if lat_delta < 0 else 'UNCHANGED'} ({lat_delta:+.2f}ms)")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description='Latte ablation: vanilla vs latent SID generator')
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--sid_generator_ckpt', type=str, required=True,
                        help='Shared checkpoint for both variants')
    parser.add_argument('--output', type=str, default='results/latte_ablation.json')

    parser.add_argument('--num_sid_tokens', type=int, default=3)
    parser.add_argument('--vocab_per_token', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--latent_token_count', type=int, default=4,
                        help='Number of latent tokens for Latte variant')
    parser.add_argument('--beam_width', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of test samples to evaluate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f'Using device: {device}')

    # 加载数据
    test_samples, item_to_sid, sid_to_items = load_test_data(
        args.test_path, args.index_path)
    trie = build_sid_trie(sid_to_items)

    # 构建并评估vanilla变体
    logger.info('Building vanilla model (no latent tokens)...')
    vanilla_model = build_model(
        'vanilla', args.sid_generator_ckpt, device,
        num_sid_tokens=args.num_sid_tokens,
        vocab_per_token=args.vocab_per_token,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        latent_token_count=0,
    )
    logger.info('Evaluating vanilla variant...')
    vanilla_metrics = evaluate_variant(
        vanilla_model, test_samples, sid_to_items, trie,
        args.beam_width, args.num_samples, device,
    )

    # 构建并评估潜在Token变体
    logger.info(f'Building latent model ({args.latent_token_count} latent tokens)...')
    latent_model = build_model(
        'latent', args.sid_generator_ckpt, device,
        num_sid_tokens=args.num_sid_tokens,
        vocab_per_token=args.vocab_per_token,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        latent_token_count=args.latent_token_count,
    )
    logger.info('Evaluating latent variant...')
    latent_metrics = evaluate_variant(
        latent_model, test_samples, sid_to_items, trie,
        args.beam_width, args.num_samples, device,
    )

    # 打印比较表
    print_ablation_table(vanilla_metrics, latent_metrics)

    # 保存结果
    results = {
        'config': {
            'num_sid_tokens': args.num_sid_tokens,
            'vocab_per_token': args.vocab_per_token,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'latent_token_count': args.latent_token_count,
            'beam_width': args.beam_width,
            'num_samples': len(test_samples[:args.num_samples]),
        },
        'vanilla': vanilla_metrics,
        'latent': latent_metrics,
        'delta': {
            k: latent_metrics.get(k, 0) - vanilla_metrics.get(k, 0)
            for k in vanilla_metrics
            if isinstance(vanilla_metrics.get(k), (int, float))
            and isinstance(latent_metrics.get(k), (int, float))
        },
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
