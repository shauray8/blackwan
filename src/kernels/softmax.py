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

