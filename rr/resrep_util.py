import torch.nn as nn
import numpy as np
from collections import defaultdict
from rr.resrep_config import ResRepConfig

#   pacesetter is not included here
def resrep_get_layer_mask_ones_and_metric_dict(model:nn.Module):
    layer_mask_ones = {}
    layer_metric_dict = {}
    report_deps = []
    for child_module in model.modules():
        # print(child_module)
        if hasattr(child_module, 'conv_idx'):

            layer_mask_ones[child_module.conv_idx] = child_module.get_num_mask_ones()
            metric_vector = child_module.get_metric_vector()
            # print('cur conv idx', child_module.conv_idx)
            # if len(metric_vector <= 512):
            #     print(metric_vector)#TODO
            for i in range(len(metric_vector)):
                layer_metric_dict[(child_module.conv_idx, i)] = metric_vector[i]
            report_deps.append(layer_mask_ones[child_module.conv_idx])
    # print('now active deps: ', report_deps)
    return layer_mask_ones, layer_metric_dict


def set_model_masks(model, layer_masked_out_filters):
    for child_module in model.modules():
        if hasattr(child_module, 'conv_idx') and child_module.conv_idx in layer_masked_out_filters:
            child_module.set_mask(layer_masked_out_filters[child_module.conv_idx])


def resrep_get_deps_and_metric_dict(origin_deps, model:nn.Module, pacesetter_dict):
    new_deps = np.array(origin_deps)
    layer_ones, metric_dict = resrep_get_layer_mask_ones_and_metric_dict(model)
    for idx, ones in layer_ones.items():
        assert ones <= origin_deps[idx]
        new_deps[idx] = ones
        # TODO include pacesetter_dict here
        if pacesetter_dict is not None:
            for follower, pacesetter in pacesetter_dict.items():
                if follower != pacesetter and pacesetter == idx:
                    new_deps[follower] = ones

    return new_deps, metric_dict


def get_cur_num_deactivated_filters(origin_deps, cur_deps, follow_dict):
    assert len(origin_deps) == len(cur_deps)
    diff = origin_deps - cur_deps
    assert np.sum(diff < 0) == 0

    result = 0
    for i in range(len(origin_deps)):
        if (follow_dict is not None) and (i in follow_dict) and (follow_dict[i] != i):
            pass
        else:
            result += origin_deps[i] - cur_deps[i]

    return result


def resrep_mask_model(origin_deps, resrep_config:ResRepConfig, model:nn.Module):
    origin_flops = resrep_config.flops_func(origin_deps)
    # print('origin flops ', origin_flops)
    cur_deps, metric_dict = resrep_get_deps_and_metric_dict(origin_deps, model,
                                                            pacesetter_dict=resrep_config.pacesetter_dict)
    # print(valve_dict)
    sorted_metric_dict = sorted(metric_dict, key=metric_dict.get)
    # print(sorted_valve_dict)
    # print(sorted_metric_dict)

    cur_flops = resrep_config.flops_func(cur_deps)
    cur_deactivated = get_cur_num_deactivated_filters(origin_deps, cur_deps, follow_dict=resrep_config.pacesetter_dict)
    # print('now deactivated {} filters'.format(cur_deactivated))
    if cur_flops > resrep_config.flops_target * origin_flops:
        next_deactivated_max = cur_deactivated + resrep_config.begin_granularity
        # print('next deac max', next_deactivated_max)
    else:
        next_deactivated_max = 9999999

    assert next_deactivated_max > 0
    attempt_deps = np.array(origin_deps)
    i = 0
    skip_idx = []
    while True:
        attempt_flops = resrep_config.flops_func(attempt_deps)
        # print('attempt flops ', attempt_flops)
        if attempt_flops <= resrep_config.flops_target * origin_flops:
            break
        attempt_layer_filter = sorted_metric_dict[i]
        if attempt_deps[attempt_layer_filter[0]] <= resrep_config.num_at_least:
            skip_idx.append(i)
            i += 1
            continue
        attempt_deps[attempt_layer_filter[0]] -= 1
        # TODO include pacesetter dict here
        if resrep_config.pacesetter_dict is not None:
            for follower, pacesetter in resrep_config.pacesetter_dict.items():
                if pacesetter == attempt_layer_filter[0] and pacesetter != follower:
                    attempt_deps[follower] -= 1
        i += 1
        if i >= next_deactivated_max:
            break

    layer_masked_out_filters = defaultdict(list)  # layer_idx : [zeros]
    for k in range(i):
        if k not in skip_idx:
            layer_masked_out_filters[sorted_metric_dict[k][0]].append(sorted_metric_dict[k][1])

    set_model_masks(model, layer_masked_out_filters)


def get_compactor_mask_dict(model:nn.Module):
    compactor_name_to_mask = {}
    compactor_name_to_kernel_param = {}
    for name, buffer in model.named_buffers():
        if 'compactor.mask' in name:
            compactor_name_to_mask[name.replace('mask', '')] = buffer
            # print(name, buffer.size())
    for name, param in model.named_parameters():
        if 'compactor.pwc.weight' in name:
            compactor_name_to_kernel_param[name.replace('pwc.weight', '')] = param
            # print(name, param.size())
    result = {}
    for name, kernel in compactor_name_to_kernel_param.items():
        mask = compactor_name_to_mask[name]
        num_filters = mask.nelement()
        if kernel.ndimension() == 4:
            if mask.ndimension() == 1:
                broadcast_mask = mask.reshape(-1, 1).repeat(1, num_filters)
                result[kernel] = broadcast_mask.reshape(num_filters, num_filters, 1, 1)
            else:
                assert mask.ndimension() == 4
                result[kernel] = mask
        else:
            assert kernel.ndimension() == 1
            result[kernel] = mask
    return result



def get_deps_if_prune_low_metric(origin_deps, model, threshold, pacesetter_dict):
    cur_deps = np.array(origin_deps)
    for child_module in model.modules():
        if hasattr(child_module, 'conv_idx'):
            metric_vector = child_module.get_metric_vector()
            num_filters_under_thres = np.sum(metric_vector < threshold)
            cur_deps[child_module.conv_idx] -= num_filters_under_thres
            cur_deps[child_module.conv_idx] = max(1, cur_deps[child_module.conv_idx])   #TODO
            #TODO pacesetter?
            if pacesetter_dict is not None:
                for follower, pacesetter in pacesetter_dict.items():
                    if pacesetter == child_module.conv_idx and pacesetter != follower:
                        cur_deps[follower] -= num_filters_under_thres
                        cur_deps[follower] = max(1, cur_deps[follower])
    return cur_deps


def resrep_get_unmasked_deps(origin_deps, model:nn.Module, pacesetter_dict):
    unmasked_deps = np.array(origin_deps)
    for child_module in model.modules():
        if hasattr(child_module, 'conv_idx'):
            layer_ones = child_module.get_num_mask_ones()
            unmasked_deps[child_module.conv_idx] = layer_ones
            # print('cur conv, ', child_module.conv_idx, 'dict is ', pacesetter_dict)
            if pacesetter_dict is not None:
                for follower, pacesetter in pacesetter_dict.items():
                    if pacesetter == child_module.conv_idx:
                        unmasked_deps[follower] = layer_ones
                    # print('cur conv ', child_module.conv_idx, 'follower is ', follower)
    return unmasked_deps

