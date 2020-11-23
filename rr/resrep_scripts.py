import numpy as np
from constants import *

def get_con_flops(input_deps, output_deps, h, w=None, kernel_size=3, groups=1):
    if w is None:
        w = h
    return input_deps * output_deps * h * w * kernel_size * kernel_size // groups

def calculate_mi1_flops(deps):
    assert len(deps) == 27
    for i in range(13):
        assert deps[2*i] == deps[2*i+1]
    result = []
    in_channels = 3
    out_h = 112
    for layer_idx in range(27):
        groups = 1 if layer_idx % 2 == 0 else deps[layer_idx]
        kernel_size = 3 if (layer_idx == 0 or layer_idx % 2 == 1) else 1
        if layer_idx in [3, 7, 11, 23]:
            out_h = out_h // 2
        result.append(get_con_flops(in_channels, deps[layer_idx], h=out_h, kernel_size=kernel_size, groups=groups))
        in_channels = deps[layer_idx]
    result.append(1000 * deps[-1])
    return np.sum(np.array(result, dtype=np.float32))

def calculate_resnet_bottleneck_flops(fd, resnet_n, original_version=False):
    num_blocks = resnet_n_to_num_blocks[resnet_n]
    d = convert_resnet_bottleneck_deps(fd)
    result = []
    # conv1
    result.append(get_con_flops(3, d[0], 112, 112, kernel_size=7))
    last_dep = d[0]
    # stage 2
    result.append(get_con_flops(last_dep, d[1][0][2], 56, kernel_size=1))
    for i in range(num_blocks[0]):
        result.append(get_con_flops(last_dep, d[1][i][0], 56, kernel_size=1))
        result.append(get_con_flops(d[1][i][0], d[1][i][1], 56, kernel_size=3))
        result.append(get_con_flops(d[1][i][1], d[1][i][2], 56, kernel_size=1))
        last_dep = d[1][i][2]
    # stage 3
    result.append(get_con_flops(last_dep, d[2][0][2], 28, kernel_size=1))
    for i in range(num_blocks[1]):
        result.append(get_con_flops(last_dep, d[2][i][0], 28 if original_version or i > 0 else 56, kernel_size=1))
        result.append(get_con_flops(d[2][i][0], d[2][i][1], 28, kernel_size=3))
        result.append(get_con_flops(d[2][i][1], d[2][i][2], 28, kernel_size=1))
        last_dep = d[2][i][2]
    # stage 4
    result.append(get_con_flops(last_dep, d[3][0][2], 14, kernel_size=1))
    for i in range(num_blocks[2]):
        result.append(get_con_flops(last_dep, d[3][i][0], 14 if original_version or i > 0 else 28, kernel_size=1))
        result.append(get_con_flops(d[3][i][0], d[3][i][1], 14, kernel_size=3))
        result.append(get_con_flops(d[3][i][1], d[3][i][2], 14, kernel_size=1))
        last_dep = d[3][i][2]
    # stage 5
    result.append(get_con_flops(last_dep, d[4][0][2], 7, kernel_size=1))
    for i in range(num_blocks[3]):
        result.append(get_con_flops(last_dep, d[4][i][0], 7 if original_version or i > 0 else 14, kernel_size=1))
        result.append(get_con_flops(d[4][i][0], d[4][i][1], 7, kernel_size=3))
        result.append(get_con_flops(d[4][i][1], d[4][i][2], 7, kernel_size=1))
        last_dep = d[4][i][2]

    # fc
    result.append(1000 * last_dep)
    return np.sum(np.array(result, dtype=np.float32))

#   fd : flattened deps
def calculate_resnet_50_flops(fd):
    return calculate_resnet_bottleneck_flops(fd, 50)

def calculate_rc_flops(deps, rc_n):
    result = []
    result.append(get_con_flops(3, deps[0], 32, 32))
    for i in range(rc_n):
        result.append(get_con_flops(deps[2*i], deps[2*i+1], 32, 32))
        result.append(get_con_flops(deps[2*i+1], deps[2*i+2], 32, 32))

    project_layer_idx = 2 * rc_n + 1
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx], 16, 16, 2))
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx + 1], 16, 16))
    result.append(get_con_flops(deps[project_layer_idx + 1], deps[project_layer_idx + 2], 16, 16))
    for i in range(rc_n - 1):
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 2], deps[2 * i + project_layer_idx + 3], 16, 16))
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 3], deps[2 * i + project_layer_idx + 4], 16, 16))

    project_layer_idx += 2 * rc_n + 1
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx], 8, 8, 2))
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx + 1], 8, 8))
    result.append(get_con_flops(deps[project_layer_idx + 1], deps[project_layer_idx + 2], 8, 8))
    for i in range(rc_n - 1):
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 2], deps[2 * i + project_layer_idx + 3], 8, 8))
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 3], deps[2 * i + project_layer_idx + 4], 8, 8))

    result.append(10*deps[-1])
    return np.sum(result)

def calculate_rc56_flops(deps):
    return calculate_rc_flops(deps, 9)
def calculate_rc110_flops(deps):
    return calculate_rc_flops(deps, 18)
