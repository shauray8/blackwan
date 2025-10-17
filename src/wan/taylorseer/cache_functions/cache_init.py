def cache_init(self, num_steps=50):   
    '''
    FIXED: Complete initialization for cache with all required keys, without calling cal_type.
    '''
    cache_dic = {}
    cache = {}
    cache[-1] = {}
    cache[-1]['cond_stream'] = {}
    cache[-1]['uncond_stream'] = {}
    cache_dic['cache_counter'] = 0

    for j in range(self.num_layers):
        cache[-1]['cond_stream'][j] = {}
        cache[-1]['uncond_stream'][j] = {}
        
        # ADD THE MISSING ATTENTION MODULE KEYS FOR ALL LAYERS
        for stream_type in ['cond_stream', 'uncond_stream']:
            cache[-1][stream_type][j]['self-attention'] = {}
            cache[-1][stream_type][j]['cross-attention'] = {}
            cache[-1][stream_type][j]['ffn'] = {}

    cache_dic['taylor_cache'] = False
    cache_dic['Delta-DiT'] = False

    # ADD MISSING KEYS
    cache_dic['cal_threshold'] = 5

    cache_dic['cache_type'] = 'random'
    cache_dic['fresh_ratio_schedule'] = 'ToCa' 
    cache_dic['fresh_ratio'] = 0.0
    cache_dic['fresh_threshold'] = 1
    cache_dic['force_fresh'] = 'global'

    mode = 'Taylor'

    if mode == 'original':
        cache_dic['cache'] = cache
        cache_dic['force_fresh'] = 'global'
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3
        
    elif mode == 'ToCa':
        cache_dic['cache_type'] = 'attention'
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.1
        cache_dic['fresh_threshold'] = 5
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3
    
    elif mode == 'Taylor':
        cache_dic['cache'] = cache
        cache_dic['fresh_threshold'] = 5
        cache_dic['cal_threshold'] = 5
        cache_dic['taylor_cache'] = True
        cache_dic['max_order'] = 1
        cache_dic['first_enhance'] = 1

    elif mode == 'Delta':
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 3
        cache_dic['Delta-DiT'] = True
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1

    # CURRENT INFO - MAKE SURE ALL KEYS EXIST
    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = num_steps
    current['stream'] = 'cond_stream'
    current['layer'] = 0
    current['module'] = 'self-attention'
    current['type'] = 'full'

    # Ensure the function returns the tuple
    result = cache_dic, current
    return result  # This MUST return the tuple