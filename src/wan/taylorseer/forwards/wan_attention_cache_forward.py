import torch 
import torch.cuda.amp as amp
from typing import Dict
from ..taylorseer_utils import taylor_cache_init, derivative_approximation, taylor_formula

def wan_attention_cache_forward(sa_dict:Dict, ca_dict:Dict, ffn_dict:Dict, e:tuple, x:torch.Tensor, distance:int):
    print(f"Step: {current['step'] if 'current' in locals() else 'unknown'}, Distance: {distance}")
    print(f"x shape: {x.shape}")
    print(f"e[2] shape: {e[2].shape if hasattr(e[2], 'shape') else type(e[2])}")
    print(f"e[5] shape: {e[5].shape if hasattr(e[5], 'shape') else type(e[5])}")
    
    seer_sa = taylor_formula(derivative_dict=sa_dict, distance=distance)
    seer_ca = taylor_formula(derivative_dict=ca_dict, distance=distance)
    seer_ffn = taylor_formula(derivative_dict=ffn_dict, distance=distance)
    
    print(f"seer_sa shape: {seer_sa.shape if hasattr(seer_sa, 'shape') else type(seer_sa)}")
    print(f"seer_ca shape: {seer_ca.shape if hasattr(seer_ca, 'shape') else type(seer_ca)}")
    print(f"seer_ffn shape: {seer_ffn.shape if hasattr(seer_ffn, 'shape') else type(seer_ffn)}")

    x = cache_add(x, seer_ca, seer_sa, seer_ffn, e)

    return x

def cache_add(x, sa, ca, ffn, e):
    """
    FIXED: Squeeze extra dimensions from e[2] and e[5]
    """
    # Squeeze the extra dimension from e[2] and e[5]
    e2_squeezed = e[2].squeeze(2)  # [1, 32760, 1, 5120] -> [1, 32760, 5120]
    e5_squeezed = e[5].squeeze(2)  # [1, 32760, 1, 5120] -> [1, 32760, 5120]
    
    # Now shapes are compatible: x: [1, 32760, 5120], e2_squeezed: [1, 32760, 5120]
    x = x + sa * e2_squeezed
    x = x + ca
    x = x + ffn * e5_squeezed
    
    return x