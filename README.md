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

3. (preprint, 2020) **State-of-the-art** channel pruning\
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
