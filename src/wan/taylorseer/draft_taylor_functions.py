# Draft formula for taylorseer (has a lot of hard limitations and does not really work)
from typing import Dict 
import torch
import math

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    if len(current['activated_steps']) < 2:
        updated_taylor_factors = {0: feature}
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors
        return
    
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    
    if abs(difference_distance) < 1e-6:
        difference_distance = 1.0
    
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        prev_key = i
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(prev_key, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            # Prevent large divisions that cause tensor explosion
            diff = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i])
            updated_taylor_factors[i + 1] = diff / max(abs(difference_distance), 1.0)  # Prevent division by small numbers
        else:
            break
    
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

def taylor_formula(derivative_dict: Dict, distance: int) -> torch.Tensor:
    if not derivative_dict or 0 not in derivative_dict:
        return torch.zeros(1) if derivative_dict else 0
    
    base_tensor = derivative_dict[0]
    output = base_tensor.clone()  
    
    for i in range(1, min(len(derivative_dict), 2)):  # LIMIT TO FIRST ORDER (i=1)
        if i not in derivative_dict:
            continue
        
        clamped_distance = max(-10, min(10, distance))  
        term_factor = (1 / math.factorial(i)) * (clamped_distance ** i)
        scaled_term = derivative_dict[i] * term_factor
        output = output + scaled_term
    
    return output

def taylor_cache_init(cache_dic: Dict, current: Dict):
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}
