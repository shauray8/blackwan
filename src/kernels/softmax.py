import math
import torch
from functools import partial
from typing import Optional, Tuple, Type, Union, Callable

import cuda.bindings.driver as cuda
import operator

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op

from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.runtime import from_dlpack


@torch.library.custom_op("quack::_softmax_fwd", mutates_args={"out"})
def _softmax_fwd(x: torch.Tensor, out: torch.Tensor) -> None:
    """Softmax forward pass.
    Args:
        x: Input tensor of shape (M, N)
    Returns:
        Softmax output tensor of same shape as x
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    N = x.size(1)
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    x_tensor, out_tensor = [convert_from_dlpack(tensor) for tensor in (x, out)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N)
    if compile_key not in _softmax_fwd.compile_cache:
        softmax_op = Softmax(dtype, N)
        _softmax_fwd.compile_cache[compile_key] = cute.compile(
            softmax_op, x_tensor, out_tensor, current_stream
        )
    _softmax_fwd.compile_cache[compile_key](x_tensor, out_tensor, current_stream)



