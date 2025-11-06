# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pandas as pd
from tqdm import tqdm

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
)
from torchao.utils import is_sm_at_least_90

assert is_sm_at_least_90(), "Hopper (SM90+) required"

DEVICE = "cuda"
BLOCK_SIZE = 128
H100_PEAK_BANDWIDTH_GBPS = 3350.0  # GB/s


def benchmark_kernel(fn, input_tensor, block_size=128, num_warmup=20, num_iter=100):
    """Benchmark a quantization kernel using CUDA events."""
    # Ensure contiguous
    input_tensor = input_tensor.contiguous()

    # Warmup (triggers Triton compilation)
    for _ in range(num_warmup):
        fn(input_tensor, block_size)
    torch.cuda.synchronize()

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iter):
        fn(input_tensor, block_size)
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / num_iter
    return avg_ms


def estimate_bandwidth(kernel_name, shape, avg_ms, block_size=128):
    m, k = shape
    # Input: bf16 (2B/element)
    input_bytes = m * k * 2

    # Output: fp8 (1B/element) + scales (4B each)
    if kernel_name in ("act_quant_lhs", "act_quant_transposed_lhs"):
        num_scales = (m + block_size - 1) // block_size
    elif kernel_name == "act_quant_rhs":
        num_scales = (k + block_size - 1) // block_size
    elif "weight" in kernel_name:
        num_row_blocks = (m + block_size - 1) // block_size
        num_col_blocks = (k + block_size - 1) // block_size
        num_scales = num_row_blocks * num_col_blocks
    else:
        num_scales = 0

    output_bytes = m * k * 1 + num_scales * 4
    total_bytes = input_bytes + output_bytes
    bandwidth_gbps = total_bytes / (avg_ms * 1e-3) / 1e9
    pct_peak = (bandwidth_gbps / H100_PEAK_BANDWIDTH_GBPS) * 100
    return bandwidth_gbps, pct_peak


def run_benchmark():
    results = []

    # Test shapes (all divisible by 128)
    test_shapes = [
        (1024, 4096),
        (4096, 4096),
        (4096, 14336),
    ]

    kernels = [
        ("act_quant_lhs", triton_fp8_blockwise_act_quant_lhs),
        ("act_quant_transposed_lhs", triton_fp8_blockwise_act_quant_transposed_lhs),
        ("act_quant_rhs", triton_fp8_blockwise_act_quant_rhs),
        ("weight_quant_rhs", triton_fp8_blockwise_weight_quant_rhs),
        ("weight_quant_transposed_rhs", triton_fp8_blockwise_weight_quant_transposed_rhs),
    ]

    for m, k in tqdm(test_shapes, desc="Shapes"):
        x = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)
        w = torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE)

        for name, fn in kernels:
            tensor = w if "weight" in name else x
            avg_ms = benchmark_kernel(fn, tensor, BLOCK_SIZE)
            bw, pct = estimate_bandwidth(name, (m, k), avg_ms, BLOCK_SIZE)

            results.append({
                "kernel": name,
                "shape": f"({m}, {k})",
                "latency_ms": round(avg_ms, 3),
                "bandwidth_gbps": round(bw, 1),
                "pct_peak_bandwidth": round(pct, 1),
            })

    return results


if __name__ == "__main__":
    results = run_benchmark()
    df = pd.DataFrame(results)
    df.to_csv("fp8_quant_benchmarks_corrected.csv", index=False)
    print(df.to_markdown(index=False))

    print(f"\nH100 peak bandwidth: {H100_PEAK_BANDWIDTH_GBPS} GB/s")
    print("Target: ≥ 80% peak → ≥ 2680 GB/s")
