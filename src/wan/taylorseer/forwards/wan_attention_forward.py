import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from wan.modules import WanModel
from wan.modules.model import sinusoidal_embedding_1d, WanAttentionBlock
from wan.modules.attention import attention

from wan.taylorseer.taylorseer_utils import taylor_cache_init, derivative_approximation, taylor_formula
from .wan_attention_cache_forward import wan_attention_cache_forward

def wan_attention_forward(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
    cache_dic,
    current
):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, L1, 6, C]
        seq_lens(Tensor): Shape [B], length of each sequence in batch
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    assert e.dtype == torch.float32
    with torch.amp.autocast('cuda', dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
    assert e[0].dtype == torch.float32

    if current['type'] == 'full':
        # This is a fresh calculation, initialize the module
        current['module'] = 'self-attention'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2), seq_lens, grid_sizes, freqs)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=y)
        
        current['module'] = 'cross-attention' 
        taylor_cache_init(cache_dic=cache_dic, current=current)
        
        y = self.cross_attn(
            self.norm2(x).float() * (1 + e[3].squeeze(2)) + e[2].squeeze(2),
            context.float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2), context_lens)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=y)
        
        current['module'] = 'ffn'
        taylor_cache_init(cache_dic=cache_dic, current=current)
        
        y = self.ffn(self.norm3(x).float() * (1 + e[5].squeeze(2)) + e[4].squeeze(2))
        derivative_approximation(cache_dic=cache_dic, current=current, feature=y)
        
        return y * e[5].squeeze(2)
    
    elif current['type'] == 'Taylor':
        # Make sure cache structure exists for all modules
        if 'self-attention' not in cache_dic['cache'][-1][current['stream']][current['layer']]:
            # Initialize missing cache entries
            cache_dic['cache'][-1][current['stream']][current['layer']]['self-attention'] = {}
            cache_dic['cache'][-1][current['stream']][current['layer']]['cross-attention'] = {}
            cache_dic['cache'][-1][current['stream']][current['layer']]['ffn'] = {}
        
        x = wan_attention_cache_forward(
            sa_dict=cache_dic['cache'][-1][current['stream']][current['layer']]['self-attention'],
            ca_dict=cache_dic['cache'][-1][current['stream']][current['layer']]['cross-attention'],
            ffn_dict=cache_dic['cache'][-1][current['stream']][current['layer']]['ffn'],
            e=e,
            x=x,
            distance= current['step'] - current['activated_steps'][-1]
        )
            
    else:
        raise ValueError(f"Not supported type: {current['type']}")

    # cross-attention & ffn function
    def cross_attn_ffn(x, context, context_lens, e):
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(
            self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
        return x

    return x