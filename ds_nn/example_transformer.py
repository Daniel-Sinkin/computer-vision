"""
danielsinkin97@gmail.com

Implementation of the original 2017 *Attention Is All You Need* Transformer
architecture (only the encoder part) with optional QKV-fused projections
for faster attention. The module can be executed as a script to benchmark
a single forward pass under various back-ends (CPU, CUDA, Apple MPS) while
collecting profiler statistics.
"""

import argparse
import time
from enum import Enum, auto

import torch
import torch.profiler

from src.transformer import Transformer

from src.common import Configs


class ProfilerMode(Enum):
    """
    Execution/Profiling modes used by :pyfunc:`main`.
    * NONE - No profiler, wall-clock measurement only (fastest).
    * CPU_ONLY - Record CPU activities, including tensor shapes & memory.
    * CPU_AND_CUDA - Record CPU **and** CUDA kernels (adds GPU timing).
    """

    NONE = auto()
    CPU_ONLY = auto()
    CPU_AND_CUDA = auto()


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
    """
    Example that creates a Transformer and profiles one forward pass

    Processes CLI args that determine what backend and what datatype should be used.
    """
    args = parse_args()

    Configs.print()

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
