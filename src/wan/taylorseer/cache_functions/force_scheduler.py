import torch
def force_scheduler(cache_dic, current):
    if cache_dic['fresh_ratio'] == 0:
        linear_step_weight = 0.0
    else: 
        linear_step_weight = 0.0
    step_factor = torch.tensor(1 - linear_step_weight + 2 * linear_step_weight * current['step'] / current['num_steps'])
    threshold = torch.round(cache_dic['fresh_threshold'] / step_factor)
    cache_dic['cal_threshold'] = threshold
