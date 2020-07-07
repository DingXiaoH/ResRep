from base_model.mobilenetv1 import *
from base_model.stagewise_resnet import *
from base_model.vgg import *
from base_model.lenet5 import create_lenet5bn

IMAGENET_STANDARD_MODEL_MAP = {
'sres50': create_SResNet50,
'smi1': create_MobileV1Imagenet,
}

CIFAR10_MODEL_MAP = {
    'src56':create_SRC56,
    'src110':create_SRC110,
    'vc':create_vc,
}

MNIST_MODEL_MAP = {
    'lenet5bn': create_lenet5bn,
}

DATASET_TO_MODEL_MAP = {
    'imagenet_standard': IMAGENET_STANDARD_MODEL_MAP,
    'cifar10': CIFAR10_MODEL_MAP,
    'mnist': MNIST_MODEL_MAP
}


#   return the model creation function
def get_model_fn(dataset_name, model_name):
    # print(DATASET_TO_MODEL_MAP[dataset_name.replace('_blank', '_standard')].keys())
    return DATASET_TO_MODEL_MAP[dataset_name.replace('_blank', '_standard')][model_name]

def get_dataset_name_by_model_name(model_name):
    for dataset_name, model_map in DATASET_TO_MODEL_MAP.items():
        if model_name in model_map:
            return dataset_name
    return None
