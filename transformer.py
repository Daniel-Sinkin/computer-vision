"""
danielsinkin97@gmail.com

Contains an implementation of the 2017 Attention is all you need Transformer,
uses QKV fusing.
"""

from typing import cast
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch.profiler


@dataclass(frozen=True)
class Configs:
    """Contains all configurations"""

    use_fused_qkv: bool = True

    asserts_enabled: bool = True


def assert_shape(x: Tensor, expected_shape: torch.Size | tuple[int, ...]) -> None:
    """Wrapper around shape assertion that is more readable"""
    if Configs.asserts_enabled:
        assert x.shape == expected_shape, f"{x.shape=} != {expected_shape=}"


def assert_same_shape(x: Tensor, y: Tensor) -> None:
    """Check that the shape of the two tensors is the same"""
    if Configs.asserts_enabled:
        assert x.shape == y.shape, f"{x.shape}!={y.shape}"


# For shape asserts so we have no magic numbers floating around
SHAPE_BROADCAST = 1


class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 12):
        super().__init__()  # type: ignore
        assert d_model % n_head == 0
        "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_h = d_model // n_head

        if Configs.use_fused_qkv:
            self.W_QKV = nn.Linear(d_model, 3 * d_model)
            self.W_Q = None
            self.W_K = None
            self.W_V = None
        else:
            self.W_QKV = None
            self.W_Q = nn.Linear(d_model, d_model)
            self.W_K = nn.Linear(d_model, d_model)
            self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        seq_len, d_model_input = x.shape
        assert d_model_input == self.d_model, f"{d_model_input=} != {self.d_model=}"

        if Configs.use_fused_qkv:
            assert all(x is None for x in (self.W_Q, self.W_K, self.W_V))
            assert self.W_QKV is not None
            qkv: Tensor = self.W_QKV(x)
            queries, keys, values = qkv.chunk(3, dim=-1)
        else:
            assert self.W_Q is not None
            assert self.W_K is not None
            assert self.W_V is not None
            assert self.W_QKV is None
            queries: Tensor = self.W_Q(x)
            keys: Tensor = self.W_K(x)
            values: Tensor = self.W_V(x)

        assert_shape(queries, (seq_len, self.d_model))
        assert_shape(keys, (seq_len, self.d_model))
        assert_shape(values, (seq_len, self.d_model))

        queries = queries.view(seq_len, self.n_head, self.d_h)
        keys = keys.view(seq_len, self.n_head, self.d_h)
        values = values.view(seq_len, self.n_head, self.d_h)

        assert_shape(queries, (seq_len, self.n_head, self.d_h))
        assert_shape(keys, (seq_len, self.n_head, self.d_h))
        assert_shape(values, (seq_len, self.n_head, self.d_h))

        queries: Tensor = queries.transpose(0, 1)
        keys: Tensor = keys.transpose(0, 1)
        values: Tensor = values.transpose(0, 1)

        assert_shape(queries, (self.n_head, seq_len, self.d_h))
        assert_shape(keys, (self.n_head, seq_len, self.d_h))
        assert_shape(values, (self.n_head, seq_len, self.d_h))

        similarity = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_h)
        assert_shape(similarity, (self.n_head, seq_len, seq_len))

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=similarity.device)
        ).unsqueeze(0)
        neg_inf = torch.finfo(similarity.dtype).min
        similarity = torch.where(causal_mask, similarity, neg_inf)

        if Configs.asserts_enabled:  # expensive setup for assert
            assert_shape(causal_mask, (SHAPE_BROADCAST, seq_len, seq_len))
            check_mask = ~causal_mask
            assert_shape(check_mask, causal_mask.shape)
            masked_values = similarity[check_mask.expand_as(similarity)]
            expected = torch.full_like(masked_values, neg_inf)
            assert torch.all(
                torch.isclose(masked_values, expected)
            ), "Causal mask failed"

        attention_weights = F.softmax(similarity, dim=-1)
        assert_shape(attention_weights, (self.n_head, seq_len, seq_len))

        attention_output = torch.matmul(attention_weights, values)
        assert_shape(attention_output, (self.n_head, seq_len, self.d_h))

        attention_output = (
            attention_output.transpose(0, 1).contiguous().view(seq_len, self.d_model)
        )
        assert_shape(attention_output, (seq_len, self.d_model))

        output: Tensor = self.W_O(attention_output)
        assert_shape(output, (seq_len, self.d_model))

        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()  # type: ignore

        self.ln_mhsa = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)

        self.mhsa = MaskedMultiHeadSelfAttention(d_model=d_model, n_head=n_head)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        output = x + cast(Tensor, self.mhsa(self.ln_mhsa(x)))
        output = output + cast(Tensor, self.feed_forward(self.ln_ff(output)))
        return output


class Transformer(nn.Module):
    def __init__(
        self, d_model: int = 768, n_head: int = 12, d_ff: int = 2024, n_layer: int = 12
    ):
        super().__init__()  # type: ignore

        self.d_model = 768
        self.n_head = 12
        self.n_layer = 12
        self.d_ff = d_ff

        self.transformer_blocks = nn.Sequential(
            *(
                TransformerBlock(d_model=self.d_model, n_head=n_head, d_ff=self.d_ff)
                for _ in range(self.n_layer)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer_blocks(x)


def main() -> None:
    trafo = Transformer()
    x = torch.randn(16, 768)

    torch.set_num_threads(1)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        trafo(x)

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


if __name__ == __main__:
    main()
