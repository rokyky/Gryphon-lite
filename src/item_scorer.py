"""
Gryphon风格的物品级评分器，用于候选重排序（任务3.2、3.3）。

DotProductScorer：用户和物品嵌入之间的简单点积。
MLPScorer：在用户、物品和SID嵌入上使用MLP，带有可选特征。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ItemScorerConfig:
    """物品评分器的配置。

    Attributes:
        scorer_type: "dotproduct"或"mlp"。
        user_dim: 用户嵌入的维度。
        item_dim: 物品嵌入的维度。
        sid_dim: SID嵌入的维度（Token嵌入之和）。
        hidden_dims: MLP评分器的隐藏层维度。
        dropout: MLP评分器的Dropout。
        activation: 激活函数（relu, gelu, tanh）。
        use_category_feature: 是否包含类别匹配特征。
        use_popularity_feature: 是否包含流行度特征。
        use_recency_feature: 是否包含新近度特征。
    """
    scorer_type: str = "mlp"
    user_dim: int = 128
    item_dim: int = 128
    sid_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    activation: str = "relu"
    use_category_feature: bool = False
    use_popularity_feature: bool = False
    use_recency_feature: bool = False


class DotProductScorer(nn.Module):
    """简单的点积评分器（任务3.2）。

    score = dot(user_emb, item_emb)

    用户嵌入：历史物品嵌入的平均池化。
    物品嵌入：来自物品元数据或学习查找表。
    """

    def __init__(
        self,
        user_dim: int = 128,
        item_dim: int = 128,
        num_items: int = 0,
        use_learned_item_emb: bool = True,
    ):
        super().__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.use_learned_item_emb = use_learned_item_emb

        if use_learned_item_emb and num_items > 0:
            self.item_embedding = nn.Embedding(num_items, item_dim)
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        else:
            self.item_embedding = None

        # 如果维度不同则进行投影
        if user_dim != item_dim:
            self.user_proj = nn.Linear(user_dim, item_dim)
        else:
            self.user_proj = nn.Identity()

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算点积分数。

        Args:
            user_embeddings: (batch, user_dim) 或 (batch, history_len, user_dim)。
            item_embeddings: (batch, num_candidates, item_dim) 或 None。
            item_ids: 可选 (batch, num_candidates) 用于学习到的嵌入。

        Returns:
            torch.Tensor: (batch, num_candidates) 分数。
        """
        # 如果需要，对用户历史进行平均池化
        if user_embeddings.dim() == 3:
            user_emb = user_embeddings.mean(dim=1)  # (batch, user_dim)
        else:
            user_emb = user_embeddings

        user_emb = self.user_proj(user_emb)  # (batch, item_dim)

        # 获取物品嵌入
        if item_embeddings is not None:
            item_emb = item_embeddings
        elif self.item_embedding is not None and item_ids is not None:
            item_emb = self.item_embedding(item_ids)  # (batch, num_candidates, item_dim)
        else:
            raise ValueError("Either item_embeddings or learned item_embedding is required.")

        # 点积
        scores = torch.bmm(
            item_emb, user_emb.unsqueeze(-1)
        ).squeeze(-1)  # (batch, num_candidates)

        return scores

    @torch.no_grad()
    def compute_user_embedding(
        self,
        history_item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """将用户嵌入计算为历史物品嵌入的均值。"""
        if history_item_embeddings.dim() == 3:
            return history_item_embeddings.mean(dim=1)
        return history_item_embeddings


class MLPScorer(nn.Module):
    """基于MLP的评分器（任务3.3）。

    score = MLP([user_emb, item_emb, sid_emb, optional_features])

    架构：
        - 拼接用户、物品和SID嵌入
        - 可选地添加类别匹配、流行度、新近度特征
        - 2-3层MLP，带ReLU/GELU
        - 单个输出神经元（分数）
    """

    def __init__(self, config: ItemScorerConfig, num_items: int = 0):
        super().__init__()
        self.config = config

        # 计算输入维度
        feature_dims = config.user_dim + config.item_dim + config.sid_dim
        extra_features = 0
        if config.use_category_feature:
            extra_features += 1
        if config.use_popularity_feature:
            extra_features += 1
        if config.use_recency_feature:
            extra_features += 1
        self.extra_features = extra_features
        input_dim = feature_dims + extra_features

        # 构建MLP层
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Dropout(config.dropout))
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            elif config.activation == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # 可选的学习嵌入
        if num_items > 0:
            self.item_embedding = nn.Embedding(num_items, config.item_dim)
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        else:
            self.item_embedding = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_embeddings: torch.Tensor,
        item_embeddings: Optional[torch.Tensor] = None,
        sid_embeddings: Optional[torch.Tensor] = None,
        item_ids: Optional[torch.Tensor] = None,
        extra_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算MLP分数。

        Args:
            user_embeddings: (batch, user_dim) 或 (batch, history_len, user_dim)。
            item_embeddings: (batch, num_candidates, item_dim) 或 None。
            sid_embeddings: (batch, num_candidates, sid_dim) 或 None。
            item_ids: 可选 (batch, num_candidates) 用于学习到的嵌入。
            extra_features: 可选 (batch, num_candidates, extra_feat_dim)。

        Returns:
            torch.Tensor: (batch, num_candidates) 分数。
        """
        B = user_embeddings.shape[0]

        # 如果需要，对用户历史进行平均池化
        if user_embeddings.dim() == 3:
            user_emb = user_embeddings.mean(dim=1)  # (B, user_dim)
        else:
            user_emb = user_embeddings

        # 确定num_candidates
        if item_embeddings is not None:
            num_candidates = item_embeddings.size(1)
        elif item_ids is not None:
            num_candidates = item_ids.size(1)
        else:
            num_candidates = 1

        # 将用户嵌入扩展到匹配候选数
        user_emb_expanded = user_emb.unsqueeze(1).expand(B, num_candidates, -1)

        # 获取物品嵌入
        if item_embeddings is None and self.item_embedding is not None and item_ids is not None:
            item_embeddings = self.item_embedding(item_ids)
        elif item_embeddings is None:
            item_embeddings = torch.zeros(B, num_candidates, self.config.item_dim, device=user_emb.device)

        # 如果缺少SID嵌入，使用默认值
        if sid_embeddings is None:
            sid_embeddings = torch.zeros(B, num_candidates, self.config.sid_dim, device=user_emb.device)

        # 拼接特征
        concat_list = [user_emb_expanded, item_embeddings, sid_embeddings]

        if extra_features is not None:
            concat_list.append(extra_features)

        combined = torch.cat(concat_list, dim=-1)  # (B, num_candidates, input_dim)

        # MLP前向
        scores = self.mlp(combined).squeeze(-1)  # (B, num_candidates)

        return scores


def create_item_scorer(
    config: ItemScorerConfig,
    num_items: int = 0,
) -> nn.Module:
    """工厂：按类型创建物品评分器。"""
    if config.scorer_type == "dotproduct":
        return DotProductScorer(
            user_dim=config.user_dim,
            item_dim=config.item_dim,
            num_items=num_items,
        )
    elif config.scorer_type == "mlp":
        return MLPScorer(config, num_items=num_items)
    else:
        raise ValueError(f"Unknown scorer type: {config.scorer_type}")
