from constants import *
from rr.resrep_builder import ResRepBuilder
from rr.resrep_config import ResRepConfig
from rr.resrep_train import resrep_train_main
from base_config import get_baseconfig_by_epoch
from model_map import get_dataset_name_by_model_name
from rr.resrep_scripts import *

import argparse
from ndp_test import general_test
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', default='src56')
    parser.add_argument('-c', '--conti_or_fs', default='fs')
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help='process rank on node')

    start_arg = parser.parse_args()

    network_type = start_arg.arch
    conti_or_fs = start_arg.conti_or_fs
    assert conti_or_fs in ['continue', 'fs']
    auto_continue = conti_or_fs == 'continue'
    print('auto continue: ', auto_continue)

    if network_type == 'sres50':
        weight_decay_strength = 1e-4
        batch_size = 256
        deps = RESNET50_ORIGIN_DEPS_FLATTENED
        succeeding_strategy = resnet_bottleneck_succeeding_strategy(50)
        print(succeeding_strategy)
        pacesetter_dict = resnet_bottleneck_follow_dict(50)
        init_hdf5 = 'torchvision_res50.hdf5'
        flops_func = calculate_resnet_50_flops
        target_layers = RESNET50_INTERNAL_KERNEL_IDXES
        lrs = LRSchedule(base_lr=0.01, max_epochs=180, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        resrep_config = ResRepConfig(target_layers=target_layers, succeeding_strategy=succeeding_strategy,
                                     pacesetter_dict=pacesetter_dict, lasso_strength=1e-4,
                                     flops_func=flops_func, flops_target=0.455, mask_interval=200,
                                     compactor_momentum=0.99, before_mask_iters=5*1281167//batch_size,
                                     begin_granularity=4, weight_decay_on_compactor=False, num_at_least=1)

    elif network_type == 'src56':
        weight_decay_strength = 1e-4
        batch_size = 64
        deps = rc_origin_deps_flattened(9)
        succeeding_strategy = rc_succeeding_strategy(9)
        pacesetter_dict = rc_pacesetter_dict(9)
        flops_func = calculate_rc56_flops
        init_hdf5 = 'src56_train/finish.hdf5'
        target_layers = rc_internal_layers(9)
        lrs = LRSchedule(base_lr=0.01, max_epochs=480, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=None, cosine_minimum=0)
        resrep_config = ResRepConfig(target_layers=target_layers, succeeding_strategy=succeeding_strategy,
                                     pacesetter_dict=pacesetter_dict, lasso_strength=1e-4,
                                     flops_func=flops_func, flops_target=0.471, mask_interval=200,
                                     compactor_momentum=0.99, before_mask_iters=5 * 50000 // batch_size,
                                     begin_granularity=4, weight_decay_on_compactor=False, num_at_least=1)

    else:
        raise ValueError('...')

    log_dir = 'resrep_models/{}_train'.format(network_type)

    weight_decay_bias = 0
    warmup_factor = 0

    config = get_baseconfig_by_epoch(network_type=network_type,
                                     dataset_name=get_dataset_name_by_model_name(network_type), dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=lrs.max_epochs, base_lr=lrs.base_lr, lr_epoch_boundaries=lrs.lr_epoch_boundaries, cosine_minimum=lrs.cosine_minimum,
                                     lr_decay_factor=lrs.lr_decay_factor,
                                     warmup_epochs=0, warmup_method='linear', warmup_factor=warmup_factor,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=log_dir,
                                     tb_dir=log_dir, save_weights=None, val_epoch_period=2, linear_final_lr=lrs.linear_final_lr,
                                     weight_decay_bias=weight_decay_bias, deps=deps)

    resrep_builder = ResRepBuilder(base_config=config, resrep_config=resrep_config)

    if resrep_config.weight_decay_on_compactor:
        no_l2_keywords = ['depth']
    else:
        no_l2_keywords = ['depth', 'compactor']
        
    print('######################################################')
    print('start ere, the original flops is ', flops_func(deps))
    print('######################################################')

    if not os.path.exists(os.path.join(config.output_dir,  'finish_converted.hdf5')):
        resrep_train_main(local_rank=start_arg.local_rank,
                          cfg=config, resrep_config=resrep_config, resrep_builder=resrep_builder, show_variables=True,
                          init_hdf5=init_hdf5,
                          auto_continue=auto_continue,
                          no_l2_keywords=no_l2_keywords)

    general_test(network_type=network_type,
                 weights=os.path.join(config.output_dir, 'finish_converted.hdf5'),
                 builder=ResRepBuilder(base_config=config, resrep_config=resrep_config,
                                       mode='deploy'))






