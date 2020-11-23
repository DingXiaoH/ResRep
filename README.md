# ResRep 

**State-of-the-art** channel pruning (a.k.a. filter pruning)! This repo contains the code for [Lossless CNN Channel Pruning via Gradient Resetting and Convolutional Re-parameterization](https://arxiv.org/abs/2007.03260).

This demo will show you how to
1. Reproduce 54.5% pruning ratio of ResNet-50 on ImageNet with 8 GPUs without accuracy drop.
2. Reproduce 52.9% pruning ratio of ResNet-56 on CIFAR-10 with 1 GPU:

About the environment:
1. We used torch==1.3.0, torchvision==0.4.1, CUDA==10.2, NVIDIA driver version==440.82, tensorboard==1.11.0 on a machine with eight 2080Ti GPUs. 
2. Our method does not rely on any new or deprecated features of any libraries, so there is no need to make an identical environment.
3. If you get any errors regarding tensorboard or tensorflow, you may simply delete the code related to tensorboard or SummaryWriter.

Citation:

	@article{ding2020lossless,
  	title={Lossless CNN Channel Pruning via Gradient Resetting and Convolutional Re-parameterization},
  	author={Ding, Xiaohan and Hao, Tianxiang and Liu, Ji and Han, Jungong and Guo, Yuchen and Ding, Guiguang},
  	journal={arXiv preprint arXiv:2007.03260},
  	year={2020}
	}

## Introduction

Channel pruning (a.k.a. filter pruning) aims to slim down a convolutional neural network (CNN) by reducing the width (i.e., numbers of output channels) of convolutional layers. However, as CNN's representational capacity depends on the width, doing so tends to degrade the performance. A traditional learning-based channel pruning paradigm applies a penalty on parameters to improve the robustness to pruning, but such a penalty may degrade the performance even before pruning. Inspired by the neurobiology research about the independence of remembering and forgetting, we propose to re-parameterize a CNN into the remembering parts and forgetting parts, where the former learn to maintain the performance and the latter learn for efficiency. By training the re-parameterized model using regular SGD on the former but a novel update rule with penalty gradients on the latter, we achieve structured sparsity, enabling us to equivalently convert the re-parameterized model into the original architecture with narrower layers. With our method, we can slim down a standard ResNet-50 with 76.15\% top-1 accuracy on ImageNet to a narrower one with only 45.5\% FLOPs and no accuracy drop.

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

5. Run ResRep. The pruned weights will be saved to "resrep_models/src56_train/finish_converted.hdf5" and automatically tested. You will get a final accuracy of 93.7 ~ 93.8. The results reported in our paper are average of 5 runs.
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

**State-of-the-art** channel pruning (preprint, 2020): [Lossless CNN Channel Pruning via Gradient Resetting and Convolutional Re-parameterization](https://arxiv.org/abs/2007.03260) (https://github.com/DingXiaoH/ResRep)

CNN component (ICCV 2019): [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf) (https://github.com/DingXiaoH/ACNet)

Channel pruning (CVPR 2019): [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html) (https://github.com/DingXiaoH/Centripetal-SGD)

Channel pruning (ICML 2019): [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html) (https://github.com/DingXiaoH/AOFP)

Unstructured pruning (NeurIPS 2019): [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf) (https://github.com/DingXiaoH/GSM-SGD)
