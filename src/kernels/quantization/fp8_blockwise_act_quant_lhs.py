# test_single_quant.py
import torch
from torchao.prototype.blockwise_fp8_training.kernels import triton_fp8_blockwise_act_quant_lhs
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torchao.prototype.moe_training.utils import (
    _is_column_major,
    _is_row_major,
)
quant_kernel_configs_with_groups = [
    triton.Config(
        {"NUM_GROUPS": groups},
        num_warps=warps,
        num_stages=stages,
    )
    for groups in [2, 16, 32, 64, 128]
    for warps in [2, 4, 8]
    for stages in [2, 4, 6]
]
EPS = 1e-12


## OPTIMIZED QUANT KERNEL
@triton.autotune(configs=quant_kernel_configs_with_groups, key=["K"])
@triton.jit
def triton_fp8_blockwise_act_quant_lhs_kernel_optim(
    x_ptr,
    x_stride_dim_0,
    x_stride_dim_1,
    y_ptr,
    y_stride_dim_0,
    y_stride_dim_1,
    s_ptr,
    s_stride_dim_0,
    s_stride_dim_1,
    M,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    # Load (num_groups x block_size) tile of x, where input is row major
    m_offs = pid_m * NUM_GROUPS + tl.arange(0, NUM_GROUPS)
    k_offs = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs = m_offs[:, None] * x_stride_dim_0 + k_offs[None, :] * x_stride_dim_1
    x_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    # Perform scaling
    max_fp8_e4m3 = 448.0
    min_fp8_e4m3 = -448.0

    # Scales for (1 x block_size) groups, shape will be (NUM_GROUPS, 1)
    amax = tl.clamp(tl.max(tl.abs(x), axis=1), min=EPS, max=float("inf")).to(tl.float64)
    scale = (max_fp8_e4m3 / amax).to(tl.float32)[:, None]
    y = x * scale
    y = tl.clamp(y, min=min_fp8_e4m3, max=max_fp8_e4m3).to(y_ptr.dtype.element_ty)

    # Write output to column major fomrat
    y_offs = m_offs[:, None] * y_stride_dim_0 + k_offs[None, :] * y_stride_dim_1
    y_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
    tl.store(y_ptr + y_offs, y, mask=y_mask)

    # Write reciprocal scales
    scale_offs = m_offs[:, None] * s_stride_dim_0 + pid_k * s_stride_dim_1
    tl.store(s_ptr + scale_offs, tl.div_rn(1.0, scale))


@triton_op("torchao::triton_fp8_blockwise_act_quant_lhs_optim", mutates_args={})
def triton_fp8_blockwise_act_quant_lhs_optim(
    x: torch.Tensor, block_size: int = 128, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input: row-major high-precision tensor
    Output: row-major, with reciprocal scales for (1 x block_size) groups stored in col-major.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert dtype in [
        torch.float8_e4m3fn,
    ], "dtype must be torch.float8_e4m3fn"
    M, K = x.size()
    y = torch.empty_like(x, dtype=dtype)
    # Write scales to column-major format to align with torch._scaled_mm requirements.
    #s = x.new_empty(M, K // block_size, dtype=torch.float32).as_strided(
    #    (M, K // block_size),
    #    (1, M),
    #)
    s = torch.empty(M, K // block_size, dtype=torch.float32, device=x.device)
    grid = lambda meta: (
        triton.cdiv(M, meta["NUM_GROUPS"]),
        triton.cdiv(K, meta["BLOCK_SIZE"]),
    )
    wrap_triton(triton_fp8_blockwise_act_quant_lhs_kernel_optim)[grid](
        x,
        x.stride(0),
        x.stride(1),
        y,
        y.stride(0),
        y.stride(1),
        s,
        s.stride(0),
        s.stride(1),
        M,
        K=K,
        BLOCK_SIZE=block_size,
        EPS=EPS,
    )
    return y, s

x = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")

# Warmup (compile)
for _ in range(10):
    triton_fp8_blockwise_act_quant_lhs_optim(x, 128)
torch.cuda.synchronize()

# Time manually
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    triton_fp8_blockwise_act_quant_lhs_optim(x, 128)
end.record()
torch.cuda.synchronize()

total_ms = start.elapsed_time(end)
avg_ms = total_ms / 100

print(f"Latency: {avg_ms:.3f} ms")

# Estimate bandwidth
bytes_total = x.numel() * (2 + 1) + (1024 // 128) * 4  # rough
gbps = bytes_total / (avg_ms * 1e-3) / 1e9
print(f"Bandwidth: {gbps:.1f} GB/s")

### ORIG KERNEL ###

for _ in range(10):
    triton_fp8_blockwise_act_quant_lhs(x, 128)
torch.cuda.synchronize()

# Time manually
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    triton_fp8_blockwise_act_quant_lhs(x, 128)
end.record()
torch.cuda.synchronize()

total_ms = start.elapsed_time(end)
avg_ms = total_ms / 100

print(f"Latency ORIG KERNEL: {avg_ms:.3f} ms")

# Estimate bandwidth
bytes_total = x.numel() * (2 + 1) + (1024 // 128) * 4  # rough
gbps = bytes_total / (avg_ms * 1e-3) / 1e9
print(f"Bandwidth ORIG KERNEL: {gbps:.1f} GB/s")

x_o,y_o = triton_fp8_blockwise_act_quant_lhs_optim(x, 128)
torch.cuda.synchronize()
x_t, y_t = triton_fp8_blockwise_act_quant_lhs(x, 128)
torch.cuda.synchronize()

s_orig_c = y_o # force row-major copy
s_mod_c = y_t
print("Max |s_orig - s_mod| =", (s_orig_c - s_mod_c).abs().max().item())

#torch.testing.assert_close(y, y_t)
