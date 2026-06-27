"""
SID生成器：用于生成SID Token序列的小型Transformer解码器。

支持：
    - 小型Transformer解码器（任务2.1）
    - 已见物品过滤（任务2.4）
    - 重复过滤（任务2.4）
    - Latte风格的潜在Token（任务4.1、4.2）
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SIDGeneratorConfig:
    """SID生成器模型的配置。

    Attributes:
        vocab_size_per_token: 每个SID位置唯一的Token数量。
        num_sid_tokens: 每个SID序列中的Token数量。
        max_history_len: 考虑的最大历史物品数量。
        hidden_dim: Transformer的隐藏维度。
        num_layers: Transformer解码器层数。
        num_heads: 注意力头数。
        dropout: Dropout概率。
        max_seq_len: 最大总序列长度（历史 + SID Token）。
        use_latent_tokens: 启用Latte风格的潜在Token（任务4.1）。
        latent_token_count: 潜在Token数量（默认0 = 禁用）。
        latent_token_dim: 潜在Token的维度（默认 = hidden_dim）。
    """
    vocab_size_per_token: int = 256
    num_sid_tokens: int = 3
    max_history_len: int = 50
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 512
    use_latent_tokens: bool = False
    latent_token_count: int = 0
    latent_token_dim: Optional[int] = None


class PositionalEncoding(nn.Module):
    """正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SIDGenerator(nn.Module):
    """用于SID Token生成的小型Transformer解码器。

    架构：
        - Token嵌入层（在所有SID位置共享）
        - 位置编码
        - N层Transformer解码器（因果掩码）
        - 每个SID Token位置的输出投影头

    使用潜在Token（Latte风格，任务4.1）：
        - 可学习的潜在Token在SID生成之前预置
        - 潜在Token关注历史（如果直接将它们放入具有因果掩码的序列中，
          则无需交叉注意力）
        - SID Token关注潜在Token + 历史
    """

    def __init__(self, config=None, vocab_per_token=256, num_sid_tokens=3,
                 hidden_dim=128, num_layers=3, num_heads=4, max_len=50,
                 use_latent_tokens=False, latent_token_count=0):
        """初始化SIDGenerator。

        支持三种调用约定：
            1) SIDGenerator(config=SIDGeneratorConfig(...))   -- dataclass配置
            2) SIDGenerator(num_sid_tokens=3, vocab_per_token=256, ...)  -- 单独参数
            3) SIDGenerator(3, 256)  -- 位置参数：(num_sid_tokens, vocab_per_token)
        """
        super().__init__()
        if isinstance(config, SIDGeneratorConfig):
            self.config = config
        elif isinstance(config, int) or config is None:
            # 位置参数或基于关键字的构造
            actual_num_sid = num_sid_tokens if config is None else config
            actual_vocab = vocab_per_token
            self.config = SIDGeneratorConfig(
                vocab_size_per_token=actual_vocab,
                num_sid_tokens=actual_num_sid,
                max_history_len=max_len,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                use_latent_tokens=use_latent_tokens,
                latent_token_count=latent_token_count if use_latent_tokens else 0,
            )
        else:
            raise TypeError(
                f"Expected SIDGeneratorConfig, int, or None for config, got {type(config)}"
            )
        self.hidden_dim = self.config.hidden_dim

        # Token嵌入（在所有SID位置共享）
        self.token_embedding = nn.Embedding(
            self.config.vocab_size_per_token, self.config.hidden_dim
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(self.config.hidden_dim, self.config.max_seq_len)

        # 可学习的特殊Token
        self.history_sep_token = nn.Parameter(
            torch.randn(1, 1, self.config.hidden_dim) * 0.02
        )
        self.eos_token_embed = nn.Parameter(
            torch.randn(1, 1, self.config.hidden_dim) * 0.02
        )

        # 潜在Token（任务4.1）
        self.use_latent_tokens = self.config.use_latent_tokens
        if self.use_latent_tokens and self.config.latent_token_count > 0:
            latent_dim = self.config.latent_token_dim or self.config.hidden_dim
            self.latent_tokens = nn.Parameter(
                torch.randn(1, self.config.latent_token_count, latent_dim) * 0.02
            )
            # 如果维度不匹配则进行投影
            if latent_dim != self.config.hidden_dim:
                self.latent_proj = nn.Linear(latent_dim, self.config.hidden_dim)
            else:
                self.latent_proj = nn.Identity()
        else:
            self.latent_tokens = None
            self.latent_proj = None

        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="relu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.config.num_layers)

        # 输出头：每个SID Token位置一个线性层
        self.output_heads = nn.ModuleList([
            nn.Linear(self.config.hidden_dim, self.config.vocab_size_per_token)
            for _ in range(self.config.num_sid_tokens)
        ])

        self._init_weights()

    def _init_weights(self):
        """用较小的值初始化权重。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(
        self,
        history_sids: torch.Tensor,
        target_sids: Optional[torch.Tensor] = None,
        num_latent: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            history_sids: (batch, history_len, num_sid_tokens) 历史物品SID。
            target_sids: 可选 (batch, target_len, num_sid_tokens) teacher-forcing目标。
            num_latent: 为此次前向传播覆盖latent_token_count。

        Returns:
            包含以下键的字典：
                "logits": (batch, target_len, num_sid_tokens, vocab_size_per_token)
                          如果提供了target_sids，则返回每个位置的logits。
                "latent_z": 如果使用了潜在Token，则返回潜在表示。
        """
        B, H, T = history_sids.shape
        device = history_sids.device

        # 展平历史SID：(B, H*T) Token索引
        history_flat = history_sids.reshape(B, H * T)

        # 嵌入历史Token
        history_emb = self.token_embedding(history_flat)  # (B, H*T, D)

        # 构建带分隔符的历史段
        sep = self.history_sep_token.expand(B, 1, -1)
        memory = torch.cat([history_emb, sep], dim=1)  # (B, H*T + 1, D)

        # 构建目标/查询序列
        if target_sids is not None:
            B2, L, T2 = target_sids.shape
            target_flat = target_sids.reshape(B2, L * T2)
            target_emb = self.token_embedding(target_flat)  # (B, L*T, D)
            # 为teacher forcing右移：预置起始Token（零）
            start = torch.zeros(B2, 1, self.hidden_dim, device=device)
            tgt = torch.cat([start, target_emb[:, :-1, :]], dim=1)
        else:
            # 自回归生成：从零开始
            tgt = torch.zeros(B, 1, self.hidden_dim, device=device)

        # 处理潜在Token（任务4.1/4.2）
        latent_count = num_latent if num_latent is not None else self.config.latent_token_count
        latent_emb = None
        if self.use_latent_tokens and latent_count > 0 and self.latent_tokens is not None:
            # 将潜在Token扩展到batch大小
            latent_emb = self.latent_tokens.expand(B, -1, -1)
            latent_emb = self.latent_proj(latent_emb)

            # 在目标序列之前拼接
            tgt = torch.cat([latent_emb, tgt], dim=1)

        # 应用位置编码
        tgt = self.pos_encoder(tgt)
        memory = self.pos_encoder(memory)

        # 为解码器自注意力创建因果掩码
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float("-inf"), device=device),
            diagonal=1,
        )

        # 解码器前向
        output = self.decoder(
            tgt, memory,
            tgt_mask=causal_mask,
        )  # (B, tgt_len, D)

        result = {}

        # 如果使用了潜在Token，提取其表示
        if latent_emb is not None:
            result["latent_z"] = output[:, :latent_count, :]

        # 计算每个位置的logits
        logits_list = []
        start_idx = latent_count if latent_emb is not None else 0

        for i in range(self.config.num_sid_tokens):
            pos_logits = self.output_heads[i](output[:, start_idx + i:start_idx + i + 1, :])
            logits_list.append(pos_logits)  # (B, 1, vocab)

        logits = torch.cat(logits_list, dim=1)  # (B, num_sid_tokens, vocab)

        if target_sids is not None:
            # 重塑为(B, L, num_sid_tokens, vocab)用于多个目标位置
            # 目前logits是(B, num_sid_tokens, vocab)，用于单步预测
            # 对于多步teacher forcing，需要不同处理
            result["logits"] = logits.unsqueeze(1)  # (B, 1, num_sid_tokens, vocab)
        else:
            result["logits"] = logits.unsqueeze(1)

        return result

    @torch.no_grad()
    def generate_single_step(
        self,
        history_sids: torch.Tensor,
        step_input: Optional[torch.Tensor] = None,
        num_latent: Optional[int] = None,
    ) -> torch.Tensor:
        """给定历史和可选的部分SID，生成下一个SID的概率。

        Args:
            history_sids: (batch, history_len, num_sid_tokens) 历史。
            step_input: 可选 (batch, 1, num_sid_tokens) 之前生成的SID。
            num_latent: 覆盖潜在Token数量。

        Returns:
            torch.Tensor: (batch, num_sid_tokens, vocab_size) logits。
        """
        self.eval()
        out = self.forward(
            history_sids=history_sids,
            target_sids=step_input,
            num_latent=num_latent,
        )
        return out["logits"].squeeze(1)  # (B, num_sid_tokens, vocab)

    @torch.no_grad()
    def get_next_token_logits(
        self,
        history_sids: torch.Tensor,
        current_sid_prefix: Optional[List[int]] = None,
        num_latent: Optional[int] = None,
    ) -> torch.Tensor:
        """获取下一个SID Token位置的logits。

        Args:
            history_sids: (1, history_len, num_sid_tokens) 单个用户历史。
            current_sid_prefix: 到目前为止已生成的部分SID Token。

        Returns:
            torch.Tensor: (vocab_size,) 下一个Token的logits。
        """
        B = history_sids.shape[0]

        if current_sid_prefix is None or len(current_sid_prefix) == 0:
            step_input = None
        else:
            prefix_tensor = torch.tensor(
                [[current_sid_prefix]], dtype=torch.long, device=history_sids.device
            )  # (1, 1, len(prefix))
            if prefix_tensor.size(-1) < self.config.num_sid_tokens:
                pad_len = self.config.num_sid_tokens - prefix_tensor.size(-1)
                pad = torch.zeros(1, 1, pad_len, dtype=torch.long, device=history_sids.device)
                prefix_tensor = torch.cat([prefix_tensor, pad], dim=-1)
            step_input = prefix_tensor

        logits = self.generate_single_step(history_sids, step_input, num_latent)
        pos = len(current_sid_prefix) if current_sid_prefix else 0
        pos = min(pos, self.config.num_sid_tokens - 1)
        return logits[0, pos, :]  # (vocab_size,)


# ===== 过滤工具（任务2.4） =====


def filter_seen_items(
    generated_sids: List[Tuple[int, ...]],
    user_history_sids: List[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    """移除用户已经交互过的SID。

    Args:
        generated_sids: 生成的SID元组列表。
        user_history_sids: 用户历史中的SID元组列表。

    Returns:
        移除已见SID后的过滤列表（保持顺序）。
    """
    seen = set(user_history_sids)
    return [sid for sid in generated_sids if sid not in seen]


def filter_duplicates(
    candidates: List[Tuple[Any, ...]],
) -> List[Tuple[Any, ...]]:
    """对候选列表去重，同时保持顺序。

    Args:
        candidates: (item_id, score)元组列表或仅item_id列表。

    Returns:
        保持首次出现顺序的去重列表。
    """
    seen: Set[Any] = set()
    result: List[Tuple[Any, ...]] = []
    for c in candidates:
        key = c[0] if isinstance(c, (list, tuple)) else c
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result
