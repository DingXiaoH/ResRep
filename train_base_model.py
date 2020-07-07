
from base_config import get_baseconfig_by_epoch
from constants import *
import os
from model_map import get_dataset_name_by_model_name
from ndp_train import train_main
import argparse

def train_base_model(
        local_rank,
        network_type,
        lrs,
                weight_decay_strength,
                batch_size,
                deps, auto_continue,
                init_hdf5=None,
        net=None,
dataset_name=None):

    log_dir = '{}_train'.format(network_type)

    weight_decay_bias = 0
    warmup_factor = 0

    if dataset_name is None:
        dataset_name = get_dataset_name_by_model_name(network_type)

    config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name,
                                     dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=lrs.max_epochs, base_lr=lrs.base_lr, lr_epoch_boundaries=lrs.lr_epoch_boundaries,
                                     lr_decay_factor=lrs.lr_decay_factor, cosine_minimum=lrs.cosine_minimum,
                                     warmup_epochs=0, warmup_method='linear', warmup_factor=warmup_factor,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=log_dir,
                                     tb_dir=log_dir, save_weights=None, val_epoch_period=5, linear_final_lr=lrs.linear_final_lr,
                                     weight_decay_bias=weight_decay_bias, deps=deps)

    builder = None
    trained_weights = os.path.join(log_dir, 'finish.hdf5')
    if not os.path.exists(trained_weights):
        train_main(local_rank, config, show_variables=True, convbuilder=builder, use_nesterov=False,
                   auto_continue=auto_continue, init_hdf5=init_hdf5, net=net)


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

    init_hdf5 = None
    net = None
    dataset_name = None

    if network_type == 'src56':
        weight_decay = 1e-4
        deps = rc_origin_deps_flattened(9)
        batch_size = 64
        lrs = LRSchedule(base_lr=0.1, max_epochs=240, lr_epoch_boundaries=[120, 180], lr_decay_factor=0.1,
                         linear_final_lr=None, cosine_minimum=None)


    elif network_type == 'vc':
        weight_decay = 1e-4
        deps = VGG_ORIGIN_DEPS
        batch_size = 64
        lrs = LRSchedule(base_lr=0.1, max_epochs=240, lr_epoch_boundaries=[120, 180], lr_decay_factor=0.1,
                         linear_final_lr=None, cosine_minimum=None)

    else:
        raise ValueError('???')

    train_base_model(local_rank=start_arg.local_rank,
                     lrs=lrs, network_type=network_type, weight_decay_strength=weight_decay,
                     batch_size=batch_size, deps=deps, auto_continue=auto_continue, init_hdf5=init_hdf5,
                     net=net, dataset_name=dataset_name)

