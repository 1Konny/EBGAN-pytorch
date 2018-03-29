# EBGAN : Energy-based Generative Adversarial Network
<br>

### Overview
Pytorch implementation of EBGAN([arxiv:1609.03126])
![overview](misc/figure_from_paper.PNG)

## Objectives
![objective](misc/objective.PNG)

## Architectures
![architecture](misc/architecture.PNG)

## Repelling Regularizer
![pt](misc/pulling-away-term.PNG)
<br>

### Dependencies
```
python 3.6.4
pytorch 0.3.1.post2
visdom
```
<br>

### Usage
initialize visdom.
```
python -m visdom.server
```
train using CIFAR10 dataset. checkpoint will automatically be saved in ```checkpoint/run1``` for every epoch.
```
python main.py --model_type skip_repeat --dataset cifar10 --env_name run1
```
you can load checkpoint and continue training. make sure ```--env_name``` matched to previous runs.
```
python main.py --model_type skip_repeat --dataset cifar10 --env_name run1 --load_ckpt True
```
you can check the training process.
```
localhost:8097
```
you can also train using your own dataset. make sure your dataset is appropriate for pytorch ImageFolder class. please check data directory tree below.
```
python main.py --model_type skip_repeat --dataset custom_dataset --env_name run1
```
<br>

### [data directory tree]
```
.
└── data
    └── CelebA
        └── img_align_celeba
            ├── 000001.jpg
            ├── 000002.jpg
            ├── ...
            └── 202599.jpg
    ├── custom_dataset
        └── folder1
            ├── image1.jpg
            ├── ...
    └── ...
```
<br>

### Result : CelebA(aligned, 64x64)
(you can download CelebA dataset [here])
```
python main.py --dataset CelebA --epoch 15 --batch_size 128 --PT_ratio 0.1 --m 20 --hidden_dim 256 --noise_dim 100
```
![celeba_fixed](misc/celeba_64x64.png)
<br>

### References
1. EBGAN : Energy-based Generative Adversarial Network([arxiv:1609.03126])

[arxiv:1609.03126]: http://arxiv.org/abs/1609.03126
[here]: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
[data directory tree]: http://pytorch.org/docs/master/torchvision/datasets.html?highlight=image%20folder#torchvision.datasets.ImageFolder
