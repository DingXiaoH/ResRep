# ResRep (ICCV 2021) 

**State-of-the-art** channel pruning (a.k.a. filter pruning)! This repo contains the code for [Lossless CNN Channel Pruning via Decoupling Remembering and Forgetting](https://arxiv.org/abs/2007.03260).

Update: accepted to **ICCV 2021**!

Update: released the log of the 54.5%-pruned ResNet-50 as kindly requested by several readers. (The experiments were done 15 months ago and the results are still SOTA.)

This demo will show you how to
1. Reproduce 54.5% pruning ratio of ResNet-50 on ImageNet with 8 GPUs without accuracy drop.
2. Reproduce 52.9% pruning ratio of ResNet-56 on CIFAR-10 with 1 GPU:

About the environment:
1. We used torch==1.3.0, torchvision==0.4.1, CUDA==10.2, NVIDIA driver version==440.82, tensorboard==1.11.0 on a machine with eight 2080Ti GPUs. 
2. Our method does not rely on any new or deprecated features of any libraries, so there is no need to make an identical environment.
3. If you get any errors regarding tensorboard or tensorflow, you may simply delete the code related to tensorboard or SummaryWriter.

Citation:

	@article{ding2020lossless,
  	title={Lossless CNN Channel Pruning via Decoupling Remembering and Forgetting},
  	author={Ding, Xiaohan and Hao, Tianxiang and Liu, Ji and Han, Jungong and Guo, Yuchen and Ding, Guiguang},
  	journal={arXiv preprint arXiv:2007.03260},
  	year={2020}
	}

## Introduction

We propose ResRep, a novel method for lossless channel pruning (a.k.a. filter pruning), which aims to slim down a convolutional neural network (CNN) by reducing the width (number of output channels) of convolutional layers. Inspired by the neurobiology research about the independence of remembering and forgetting, we propose to re-parameterize a CNN into the remembering parts and forgetting parts, where the former learn to maintain the performance and the latter learn for efficiency. By training the re-parameterized model using regular SGD on the former but a novel update rule with penalty gradients on the latter, we realize structured sparsity, enabling us to equivalently convert the re-parameterized model into the original architecture with narrower layers. Such a methodology distinguishes ResRep from the traditional learning-based pruning paradigm that applies a penalty on parameters to produce structured sparsity, which may suppress the parameters essential for the remembering. Our method slims down a standard ResNet-50 with 76.15% accuracy on ImageNet to a narrower one with only 45% FLOPs and no accuracy drop, which is the first to achieve lossless pruning with such a high compression ratio, to the best of our knowledge.

## Prune ResNet-50 on ImageNet with a pruning ratio of 54.5% (FLOPs)

1. Enter this directory.

2. Make a soft link to your ImageNet directory, which contains "train" and "val" directories.
```
ln -s YOUR_PATH_TO_IMAGENET imagenet_data
```

3. Set the environment variables.
```
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

4. Download the official torchvision model, rename the parameters in our namestyle, and save the weights to "torchvision_res50.hdf5".
```
python transform_torchvision.py
```

5. Run ResRep. The pruned weights will be saved to "resrep_models/sres50_train/finish_converted.hdf5" and automatically tested.
```
python -m torch.distributed.launch --nproc_per_node=8 rr/exp_resrep.py -a sres50
```

6. Show the name and shape of weights in the pruned model.
```
python display_hdf5.py resrep_models/sres50_train/finish_converted.hdf5
```

## Prune ResNet-56 on CIFAR-10 with a pruning ratio of 52.9% (FLOPs)

1. Enter this directory.

2. Make a soft link to your CIFAR-10 directory. If the dataset is not found in the directory, it will be automatically downloaded.
```
ln -s YOUR_PATH_TO_CIFAR cifar10_data
```

3. Set the environment variables.
```
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
```

4. Train the base model. The weights will be saved to "src56_train/finish.hdf5"
```
python train_base_model.py -a src56
```

5. Run ResRep. The pruned weights will be saved to "resrep_models/src56_train/finish_converted.hdf5" and automatically tested.
```
python rr/exp_resrep.py -a src56
```

6. Show the name and shape of weights in the pruned model.
```
python display_hdf5.py resrep_models/src56_train/finish_converted.hdf5
```

## How to prune your own model

***Pruning simple models or easy-to-prune layers in a complicated model***

First, let's clarify the meanings of some architecutre-specific constants and functions defined for the pruning process.

1. ```deps```, the width of every conv layer is defined by an array named "deps". For example,
```
RESNET50_ORIGIN_DEPS_FLATTENED = [64,256,64,64,256,64,64,256,64,64,256,512,128,128,512,128,128,512,128,128,512,128,128,512,
                                  1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,
                                  2048,512, 512, 2048,512, 512, 2048,512, 512, 2048]
```
Note that we build the projection (1x1 conv shortcut) layer before the parallel residual block (L61 in ```stagewise_resnet.py```), so that its width (256) preceds the widths of the three layers of the residual block (64, 64, 256).

2. ```calculate_SOME_MODEL_flops```, the function to calculate the FLOPs of a specific architecuture given the "deps". It is architecture-specific. You may follow ```calculate_resnet_bottleneck_flops``` in ```resrep_scripts.py``` to define it for your own model.

3. ```succeeding_strategy``` defines how the layers follow others. If layer B follows layer A (i.e., pruning the output channels of layer A triggers the removal of the corresponding input channels of layer B), we should have ```succeeding_strategy\[A\]=B```. This is the common case of simple models. For example, the succeeding_strategy of VGG-16 should be ```\{0:1, 1:2, 2:3, ...\}```. 

However, some layers in some complicated models are a bit tricky to prune. In the experiments reported in the paper, we only pruned the internal layers of ResNets (i.e., the first layer of every res block of Res56 and the first two layers of every res block of Res50) but did not prune the tricky layers. You may skip the following content if you do not intend to prune those layers.

***Pruning complicated models and tricky layers***

4. Complicated ```succeeding_strategy```. For example, when you prune the last layers of stage1 and stage2 in ResNet-56 (i.e., the last layer of the last res block), which are indexed 18 and 37, you need to prune the input channels of the first two layers of the next stage accordingly, so that the succeeding_strategy is ```{1: 2, 3: 4, ..., 18: [19, 20], ..., 37: [38, 39], ...}```. 

5. ```follow_dict``'. Some layers must be pruned following others (e.g., the last layers of every res block in a stage must be pruned following the projection layer of the stage). If layer A must be pruned following layer B, we define ```follow_dict\[A\]=B``'. We did not use it in our experiments because there are no such constraints if we only prune the internal layers. We discussed this problem in a CVPR-2019 paper [C-SGD](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.pdf).

The above-mentioned constants are inputs to ```compactor_convert```, which is a generic method for pruning and converting a model with compactors. However, given a specific architecture, compared to figuring out how such constants should be defined, writing a standalone pruning function for your architecture may be easier. ``compactor_convert_mi1``` is an example for pruning MobileNet-V1. You need to
1. Figure out how the layers connect to the others. After pruning a layer, prune the input channels of its every following layer correctly.
2. If some layers must be pruned following others, do that correctly.
3. Handle the depth-wise layers, BN and other custom layers correctly.

## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. (preprint, 2021) **A powerful MLP-style CNN building block**\
[RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition](https://arxiv.org/abs/2105.01883)\
[code](https://github.com/DingXiaoH/RepMLP).

2. (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **83.55%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

3. (ICCV, 2021) **State-of-the-art** channel pruning\
[Lossless CNN Channel Pruning via Decoupling Remembering and Forgetting](https://arxiv.org/abs/2007.03260)\
[code](https://github.com/DingXiaoH/ResRep).

4. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

5. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)
