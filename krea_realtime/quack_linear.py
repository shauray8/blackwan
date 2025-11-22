import torch
torch.set_grad_enabled(False)
from quack.mlp import MLP as QuackMLP

class QuackSwiGLULinear(torch.nn.Module):
    def __init__(self, hidden_features, intermediate_features, out_features, device=None, dtype=None):
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(hidden_features, 2 * intermediate_features, bias=False, device=device, dtype=dtype)
        self.down_proj = torch.nn.Linear(intermediate_features, out_features, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return QuackMLP(
            x,
            self.gate_up_proj.weight,
            self.down_proj.weight,
            activation="swiglu",
            fuse_grad_accum=False,
            tuned=False
        )
        
def pytorch_swiglu_mlp(x, gate_up_weight, down_weight):
    gate_up = torch.nn.functional.linear(x, gate_up_weight)  
    gate, up = gate_up.chunk(2, dim=-1)                      # [M, I], [M, I]
    y = torch.nn.functional.silu(gate) * up                  
    return torch.nn.functional.linear(y, down_weight)

def benchmark_mlp(
    mlp_fn,
    hidden=5120,
    intermediate=13824,
    batch_shape=(1, 3, 16, 60, 104),
    dtype=torch.bfloat16,
    device="cuda",
    num_warmup=10,
    num_iters=100
):
    torch.manual_seed(42)
    x = torch.randn(*batch_shape, hidden, device=device, dtype=dtype).reshape(-1, hidden)
    M = x.shape[0]

    gate_up_weight = torch.randn(2 * intermediate, hidden, device=device, dtype=dtype)
    down_weight = torch.randn(hidden, intermediate, device=device, dtype=dtype)

    for _ in range(num_warmup):
        _ = mlp_fn(x, gate_up_weight, down_weight)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        _ = mlp_fn(x, gate_up_weight, down_weight)
    end.record()
    torch.cuda.synchronize()
    latency_ms = start.elapsed_time(end) / num_iters

    flops = 2 * M * (2 * intermediate * hidden + hidden * intermediate)
    tflops = flops / (latency_ms * 1e-3) / 1e12
    return latency_ms, tflops

def quack_swiglu_mlp(x, gate_up_weight, down_weight, gate_up_bias=None, down_bias=None):
    mlp = QuackSwiGLULinear(
        hidden_features=gate_up_weight.shape[1],
        intermediate_features=gate_up_weight.shape[0] // 2,  
        out_features=down_weight.shape[0],            
        device=x.device,
        dtype=x.dtype
    )
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(gate_up_weight)  
        mlp.down_proj.weight.copy_(down_weight) 
    return mlp(x)

if __name__ == "__main__":
    print("Input: [1,3,16,60,104,5120] → gate_up → swiglu → down")
    print("-" * 60)
    lat_torch, tflops_torch = benchmark_mlp(pytorch_swiglu_mlp)
    print(f"PyTorch SwiGLU (unfused) : {lat_torch:.3f} ms | {tflops_torch:.1f} TFLOPs/s")
    lat_quack, tflops_quack = benchmark_mlp(quack_swiglu_mlp)
    print(f"Quack SwiGLU (fused)    : {lat_quack:.3f} ms | {tflops_quack:.1f} TFLOPs/s")
    speedup = lat_torch / lat_quack
    print("-" * 60)
    print(f"Speedup: {speedup:.2f}x")
