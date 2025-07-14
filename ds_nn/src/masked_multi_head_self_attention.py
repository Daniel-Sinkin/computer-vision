"""danielsinkin97@gmail.com"""

import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from .common import Configs, assert_same_shape, assert_shape, BROADCAST_SHAPE


class MaskedMultiHeadSelfAttention(nn.Module):
    """
    Causal masked multi-head self-attention (MSHA)
    """

    def __init__(self, d_model: int = 768, n_head: int = 12):
        super().__init__()  # type: ignore
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_h = d_model // n_head

        if Configs.use_fused_qkv:
            self.W_QKV = nn.Linear(d_model, 3 * d_model)
            self.W_Q = self.W_K = self.W_V = None
        else:
            self.W_QKV = None
            self.W_Q = nn.Linear(d_model, d_model)
            self.W_K = nn.Linear(d_model, d_model)
            self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes Scaled Dot-Product Attention
        """
        batch, seq_len, d_model_input = x.shape
        assert d_model_input == self.d_model, f"{d_model_input=} != {self.d_model=}"

        if Configs.use_fused_qkv:
            assert all(x is None for x in (self.W_Q, self.W_K, self.W_V))
            assert self.W_QKV is not None
            qkv: Tensor = self.W_QKV(x)
            assert_shape(qkv, (batch, seq_len, 3 * self.d_model))
            queries, keys, values = qkv.chunk(3, dim=-1)
        else:
            assert self.W_Q is not None
            assert self.W_K is not None
            assert self.W_V is not None
            assert self.W_QKV is None
            queries: Tensor = self.W_Q(x)
            keys: Tensor = self.W_K(x)
            values: Tensor = self.W_V(x)

        assert_shape(queries, (batch, seq_len, self.d_model))
        assert_same_shape(queries, keys)
        assert_same_shape(queries, values)

        queries = queries.view(batch, seq_len, self.n_head, self.d_h)
        keys = keys.view(batch, seq_len, self.n_head, self.d_h)
        values = values.view(batch, seq_len, self.n_head, self.d_h)

        assert_shape(queries, (batch, seq_len, self.n_head, self.d_h))
        assert_same_shape(queries, keys)
        assert_same_shape(queries, values)

        queries: Tensor = queries.permute(0, 2, 1, 3)
        keys: Tensor = keys.permute(0, 2, 1, 3)
        values: Tensor = values.permute(0, 2, 1, 3)

        assert_shape(queries, (batch, self.n_head, seq_len, self.d_h))
        assert_same_shape(queries, keys)
        assert_same_shape(queries, values)

        similarity = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_h)
        assert_shape(similarity, (batch, self.n_head, seq_len, seq_len))

        causal_mask = (
            torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=similarity.device)
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        neg_inf = torch.finfo(similarity.dtype).min
        similarity = torch.where(causal_mask, similarity, neg_inf)

        if Configs.asserts_enabled:  # expensive setup for assert
            assert_shape(
                causal_mask, (BROADCAST_SHAPE, BROADCAST_SHAPE, seq_len, seq_len)
            )
            check_mask = ~causal_mask
            assert_same_shape(check_mask, causal_mask)
            masked_values = similarity[check_mask.expand_as(similarity)]
            expected = torch.full_like(masked_values, neg_inf)
            assert torch.all(
                torch.isclose(masked_values, expected)
            ), "Causal mask failed"

        attention_weights = F.softmax(similarity, dim=-1)
        assert_shape(attention_weights, (batch, self.n_head, seq_len, seq_len))

        attention_output = torch.matmul(attention_weights, values)
        assert_shape(attention_output, (batch, self.n_head, seq_len, self.d_h))

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        assert_shape(attention_output, (batch, seq_len, self.n_head, self.d_h))

        attention_output = attention_output.view(batch, seq_len, self.d_model)
        assert_shape(attention_output, (batch, seq_len, self.d_model))

        output: Tensor = self.W_O(attention_output)
        assert_shape(output, (batch, seq_len, self.d_model))

        return output
