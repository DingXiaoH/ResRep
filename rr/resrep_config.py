from collections import namedtuple

ResRepConfig = namedtuple('ResRepConfig', ['target_layers',
                                     'succeeding_strategy',
                                     'pacesetter_dict',
                                     'lasso_strength',
                                     'flops_func',
                                     'flops_target',
                                     'mask_interval',
                                     'compactor_momentum',
                                     'before_mask_iters',
                                     'begin_granularity',
                                     'weight_decay_on_compactor',
                                     'num_at_least',
                                     ])