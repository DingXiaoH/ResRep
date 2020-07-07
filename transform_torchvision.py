import torchvision.models as models
from utils.misc import save_hdf5

def transform_res50():
    key_replace_dict = {
        'layer1.': 'stage1.block',
        'layer2.': 'stage2.block',
        'layer3.': 'stage3.block',
        'layer4.': 'stage4.block',

        'conv1.weight': 'conv1.conv.weight',
        'bn1.': 'conv1.bn.',
        'conv2.weight': 'conv2.conv.weight',
        'bn2.': 'conv2.bn.',
        'conv3.weight': 'conv3.conv.weight',
        'bn3.': 'conv3.bn.',

        '0.downsample.0.weight': 'projection.conv.weight',
        '0.downsample.1.': 'projection.bn.'
    }

    exact_replace_dict = {
        'conv1.weight': 'conv1.conv.weight',
        'bn1.weight': 'conv1.bn.weight',
        'bn1.bias': 'conv1.bn.bias',
        'bn1.running_mean': 'conv1.bn.running_mean',
        'bn1.running_var': 'conv1.bn.running_var'
    }

    def replace_keyword(origin_name):
        for a, b in key_replace_dict.items():
            if a in origin_name:
                return origin_name.replace(a, b)
        return origin_name

    resnet18 = models.resnet50(pretrained=True)

    save_dict = {}
    for k, v in resnet18.state_dict().items():
        value = v.cpu().numpy()
        if k in exact_replace_dict:
            save_dict[exact_replace_dict[k]] = value
        elif 'downsample' in k:
            save_dict[k.replace('layer', 'stage')
                .replace('0.downsample.0.weight', 'projection.conv.weight')
                .replace('0.downsample.1.', 'projection.bn.')] = value
        else:
            save_dict[replace_keyword(replace_keyword(replace_keyword(k)))] = value

    save_hdf5(save_dict, 'torchvision_res50.hdf5')


transform_res50()



