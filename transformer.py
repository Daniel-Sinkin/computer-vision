"""
danielsinkin97@gmail.com

Contains an implementation of the 2017 Attention is all you need Transformer,
uses QKV fusing.
"""

import argparse
from typing import cast
from dataclasses import dataclass
import math
import time
from enum import Enum, auto

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import torch.profiler


@dataclass(frozen=True)
class Configs:
    use_fused_qkv: bool = True
    asserts_enabled: bool = False


class ProfilerMode(Enum):
    NONE = auto()
    CPU_ONLY = auto()
    CPU_AND_CUDA = auto()


def assert_shape(x: Tensor, expected_shape: torch.Size | tuple[int, ...]) -> None:
    """Wrapper around shape assertion that is more readable"""
    if Configs.asserts_enabled:
        assert x.shape == expected_shape, f"{x.shape=} != {expected_shape=}"


def assert_same_shape(x: Tensor, y: Tensor) -> None:
    """Check that the shape of the two tensors is the same"""
    if Configs.asserts_enabled:
        assert x.shape == y.shape, f"{x.shape}!={y.shape}"


# For shape asserts so we have no magic numbers floating around
BROADCAST_SHAPE = 1


class MaskedMultiHeadSelfAttention(nn.Module):
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
        """Computes MHSA"""
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

        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_ff = d_ff

        self.transformer_blocks = nn.Sequential(
            *(
                TransformerBlock(d_model=self.d_model, n_head=n_head, d_ff=self.d_ff)
                for _ in range(self.n_layer)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer_blocks(x)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line flags that control the execution backend.

    Flags
    -----
    --mps
        Use the Apple-GPU (MPS backend) and run in float16 precision.

    --mps-f32
        Use the Apple-GPU with float32 precision.

    --cuda
        Use CUDA GPU (if available) with float16 precision.

    If none are provided, the script defaults to CPU (float32) execution
    with torch.profiler enabled.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with boolean attributes `mps`, `mps_f32`, and `cuda`.
    """
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--mps", action="store_true", help="Run on Apple-GPU with float16")
    g.add_argument(
        "--mps-f32", action="store_true", help="Run on Apple-GPU with float32"
    )
    g.add_argument("--cuda", action="store_true", help="Run on CUDA GPU with float16")
    g.add_argument(
        "--cuda-f32", action="store_true", help="Run on CUDA GPU with float32"
    )
    return parser.parse_args()


def main() -> None:
    """Example that creates a transformer and profiles one forward pass"""

    args = parse_args()

    if args.mps or args.mps_f32:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS activated but MPS backend not availiable for PyTorch."
            )
        device = torch.device("mps")
        dtype = torch.float16 if args.mps else torch.float32
        profiler_mode = ProfilerMode.NONE
        print(f"Using MPS, ({dtype})")
    elif args.cuda or args.cuda_f32:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA selected but not compatible GPU found.")
        device = torch.device("cuda")
        dtype = torch.float32 if args.cuda_f32 else torch.float16
        profiler_mode = ProfilerMode.CPU_AND_CUDA
        print(f"Using CUDA, ({dtype})")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        profiler_mode = ProfilerMode.CPU_ONLY
        print("Using CPU (float32)")

    trafo = Transformer().eval().to(device=device, dtype=dtype)
    torch.set_num_threads(torch.get_num_threads() or 8)

    batch, seq_len, d_model = 16, 512, 768
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=dtype)

    # Warmup run
    with torch.no_grad():
        for _ in range(3):
            trafo(x)

    if profiler_mode == ProfilerMode.NONE:
        assert device.type == "mps"
        torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = trafo(x)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        print(f"{device.type.upper()} forward time: {(t1 - t0) * 1000:.2f} ms")
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if profiler_mode == ProfilerMode.CPU_AND_CUDA:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            trafo(x)

        if profiler_mode == ProfilerMode.CPU_AND_CUDA:
            sort_key = "cuda_time_total"
        else:
            sort_key = "self_cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=20))  # type: ignore


if __name__ == "__main__":
    main()
