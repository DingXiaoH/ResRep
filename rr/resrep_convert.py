from rr.compactor import CompactorLayer
import numpy as np
import torch.nn.functional as F
import torch
from utils.misc import save_hdf5

def _fuse_kernel(kernel, gamma, running_var, eps):
    print('fusing: kernel shape', kernel.shape)
    std = np.sqrt(running_var + eps)
    t = gamma / std
    t = np.reshape(t, (-1, 1, 1, 1))
    print('fusing: t', t.shape)
    t = np.tile(t, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    return kernel * t

def _fuse_bias(running_mean, running_var, gamma, beta, eps, bias=None):
    if bias is None:
        return beta - running_mean * gamma / np.sqrt(running_var + eps)
    else:
        return beta + (bias - running_mean) * gamma / np.sqrt(running_var + eps)

def fuse_conv_bn(save_dict, pop_name_set, kernel_name):
    mean_name = kernel_name.replace('.conv.weight', '.bn.running_mean')
    var_name = kernel_name.replace('.conv.weight', '.bn.running_var')
    gamma_name = kernel_name.replace('.conv.weight', '.bn.weight')
    beta_name = kernel_name.replace('.conv.weight', '.bn.bias')
    pop_name_set.add(mean_name)
    pop_name_set.add(var_name)
    pop_name_set.add(gamma_name)
    pop_name_set.add(beta_name)
    mean = save_dict[mean_name]
    var = save_dict[var_name]
    gamma = save_dict[gamma_name]
    beta = save_dict[beta_name]
    kernel_value = save_dict[kernel_name]
    print('kernel name', kernel_name)
    print('kernel, mean, var, gamma, beta', kernel_value.shape, mean.shape, var.shape, gamma.shape, beta.shape)
    return _fuse_kernel(kernel_value, gamma, var, eps=1e-5), _fuse_bias(mean, var, gamma, beta, eps=1e-5)

def fold_conv(fused_k, fused_b, thresh, compactor_mat):
    metric_vec = np.sqrt(np.sum(compactor_mat ** 2, axis=(1, 2, 3)))
    filter_ids_below_thresh = np.where(metric_vec < thresh)[0]

    if len(filter_ids_below_thresh) == len(metric_vec):
        sortd_ids = np.argsort(metric_vec)
        filter_ids_below_thresh = sortd_ids[:-1]    #TODO preserve at least one filter

    if len(filter_ids_below_thresh) > 0:
        compactor_mat = np.delete(compactor_mat, filter_ids_below_thresh, axis=0)

    kernel = F.conv2d(torch.from_numpy(fused_k).permute(1, 0, 2, 3), torch.from_numpy(compactor_mat),
                      padding=(0, 0)).permute(1, 0, 2, 3)
    Dprime = compactor_mat.shape[0]
    bias = np.zeros(Dprime)
    for i in range(Dprime):
        bias[i] = fused_b.dot(compactor_mat[i,:,0,0])

    if type(bias) is not np.ndarray:
        bias = np.array([bias])

    return kernel, bias, filter_ids_below_thresh


def compactor_convert(model, origin_deps, thresh, pacesetter_dict, succ_strategy, save_path):
    compactor_mats = {}
    for submodule in model.modules():
        if hasattr(submodule, 'conv_idx'):
            compactor_mats[submodule.conv_idx] = submodule.pwc.weight.detach().cpu().numpy()

    pruned_deps = np.array(origin_deps)

    cur_conv_idx = -1
    pop_name_set = set()

    kernel_name_list = []
    save_dict = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu().numpy()
        if v.ndim in [2, 4] and 'compactor.pwc' not in k and 'align_opr.pwc' not in k:
            kernel_name_list.append(k)
        save_dict[k] = v

    for conv_id, kernel_name in enumerate(kernel_name_list):
        kernel_value = save_dict[kernel_name]
        if kernel_value.ndim == 2:
            continue
        fused_k, fused_b = fuse_conv_bn(save_dict, pop_name_set, kernel_name)
        cur_conv_idx += 1
        fold_direct = cur_conv_idx in compactor_mats
        fold_follower = (pacesetter_dict is not None and cur_conv_idx in pacesetter_dict and pacesetter_dict[cur_conv_idx] in compactor_mats)
        if fold_direct or fold_follower:
            if fold_direct:
                fm = compactor_mats[cur_conv_idx]
            else:
                fm = compactor_mats[pacesetter_dict[cur_conv_idx]]
            fused_k, fused_b, pruned_ids = fold_conv(fused_k, fused_b, thresh, fm)
            pruned_deps[cur_conv_idx] -= len(pruned_ids)
            print('pruned ids: ', pruned_ids)
            if len(pruned_ids) > 0 and conv_id in succ_strategy:
                followers = succ_strategy[conv_id]
                if type(followers) is not list:
                    followers = [followers]
                for fo in followers:
                    fo_kernel_name = kernel_name_list[fo]
                    fo_value = save_dict[fo_kernel_name]
                    if fo_value.ndim == 4:
                        fo_value = np.delete(fo_value, pruned_ids, axis=1)
                    else:
                        fc_idx_to_delete = []
                        num_filters = kernel_value.shape[0]
                        fc_neurons_per_conv_kernel = fo_value.shape[1] // num_filters
                        print('{} filters, {} neurons per kernel'.format(num_filters, fc_neurons_per_conv_kernel))
                        base = np.arange(0, fc_neurons_per_conv_kernel * num_filters, num_filters)
                        for i in pruned_ids:
                            fc_idx_to_delete.append(base + i)
                        if len(fc_idx_to_delete) > 0:
                            fo_value = np.delete(fo_value, np.concatenate(fc_idx_to_delete, axis=0), axis=1)
                    save_dict[fo_kernel_name] = fo_value

        save_dict[kernel_name] = fused_k
        save_dict[kernel_name.replace('.weight', '.bias')] = fused_b

    save_dict['deps'] = pruned_deps
    for name in pop_name_set:
        save_dict.pop(name)

    final_dict = {k.replace('module.', '') : v for k, v in save_dict.items() if 'num_batches' not in k and 'compactor' not in k}

    save_hdf5(final_dict, save_path)
    print('---------------saved {} numpy arrays to {}---------------'.format(len(save_dict), save_path))


def compactor_convert_mi1(model, origin_deps, thresh, save_path):
    compactor_mats = {}
    for submodule in model.modules():
        if hasattr(submodule, 'conv_idx'):
            compactor_mats[submodule.conv_idx] = submodule.pwc.weight.detach().cpu().numpy()

    pruned_deps = np.array(origin_deps)

    pop_name_set = set()

    kernel_name_list = []
    save_dict = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu().numpy()
        if v.ndim in [2, 4] and 'compactor.pwc' not in k and 'align_opr.pwc' not in k:
            kernel_name_list.append(k)
        save_dict[k] = v

    for conv_id, kernel_name in enumerate(kernel_name_list):
        kernel_value = save_dict[kernel_name]
        if kernel_value.ndim == 2:
            continue
        fused_k, fused_b = fuse_conv_bn(save_dict, pop_name_set, kernel_name)

        save_dict[kernel_name] = fused_k
        save_dict[kernel_name.replace('.weight', '.bias')] = fused_b

        fold_direct = conv_id in compactor_mats
        if fold_direct:
            fm = compactor_mats[conv_id]
            fused_k, fused_b, pruned_ids = fold_conv(fused_k, fused_b, thresh, fm)

            save_dict[kernel_name] = fused_k
            save_dict[kernel_name.replace('.weight', '.bias')] = fused_b

            pruned_deps[conv_id] -= len(pruned_ids)
            print('pruned ids: ', pruned_ids)
            if len(pruned_ids) == 0:
                continue
            fo_kernel_name = kernel_name_list[conv_id + 1]
            if 'linear' in fo_kernel_name:
                fo_value = save_dict[fo_kernel_name]
                fc_idx_to_delete = []
                num_filters = kernel_value.shape[0]
                fc_neurons_per_conv_kernel = fo_value.shape[1] // num_filters
                print('{} filters, {} neurons per kernel'.format(num_filters, fc_neurons_per_conv_kernel))
                base = np.arange(0, fc_neurons_per_conv_kernel * num_filters, num_filters)
                for i in pruned_ids:
                    fc_idx_to_delete.append(base + i)
                if len(fc_idx_to_delete) > 0:
                    fo_value = np.delete(fo_value, np.concatenate(fc_idx_to_delete, axis=0), axis=1)
                save_dict[fo_kernel_name] = fo_value
            else:
                #   this_layer - following_pw - following_dw
                #   adjust the beta of following_dw by the to-be-deleted channel of following_pw
                #   delete the to-be-deleted channel of following_pw
                fol_dw_kernel_name = kernel_name_list[conv_id + 1]
                fol_dw_kernel_value = save_dict[fol_dw_kernel_name]
                fol_dw_beta_name = fol_dw_kernel_name.replace('conv.weight', 'bn.bias')
                fol_dw_beta_value = save_dict[fol_dw_beta_name]

                pw_kernel_name = kernel_name_list[conv_id + 2]
                pw_kernel_value = save_dict[pw_kernel_name]
                pw_beta_name = pw_kernel_name.replace('conv.weight', 'bn.bias')
                pw_beta_value = save_dict[pw_beta_name]
                pw_var_value = save_dict[pw_kernel_name.replace('conv.weight', 'bn.running_var')]
                pw_gamma_value = save_dict[pw_kernel_name.replace('conv.weight', 'bn.weight')]

                for pri in pruned_ids:
                    compensate_beta = np.abs(fol_dw_beta_value[pri]) * (pw_kernel_value[:, pri, 0, 0] * pw_gamma_value / np.sqrt(pw_var_value + 1e-5))  # TODO because of relu
                    pw_beta_value += compensate_beta
                save_dict[pw_beta_name] = pw_beta_value
                save_dict[pw_kernel_name] = np.delete(pw_kernel_value, pruned_ids, axis=1)

                fol_dw_kernel_value = np.delete(fol_dw_kernel_value, pruned_ids, axis=0)
                fol_dw_beta_value = np.delete(fol_dw_beta_value, pruned_ids)
                save_dict[fol_dw_kernel_name] = fol_dw_kernel_value
                save_dict[fol_dw_beta_name] = fol_dw_beta_value
                fol_dw_gamma_name = fol_dw_kernel_name.replace('conv.weight', 'bn.weight')
                fol_dw_mean_name = fol_dw_kernel_name.replace('conv.weight', 'bn.running_mean')
                fol_dw_var_name = fol_dw_kernel_name.replace('conv.weight', 'bn.running_var')
                save_dict[fol_dw_gamma_name] = np.delete(save_dict[fol_dw_gamma_name], pruned_ids)
                save_dict[fol_dw_mean_name] = np.delete(save_dict[fol_dw_mean_name], pruned_ids)
                save_dict[fol_dw_var_name] = np.delete(save_dict[fol_dw_var_name], pruned_ids)
                pruned_deps[conv_id+1] -= len(pruned_ids)

    save_dict['deps'] = pruned_deps
    for name in pop_name_set:
        save_dict.pop(name)

    final_dict = {k.replace('module.', '') : v for k, v in save_dict.items() if 'num_batches' not in k and 'compactor' not in k}

    save_hdf5(final_dict, save_path)
    print('---------------saved {} numpy arrays to {}---------------'.format(len(save_dict), save_path))