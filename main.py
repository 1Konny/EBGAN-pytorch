"""main.py"""

import argparse

import numpy as np
import torch

from solver import EBGAN
from utils import str2bool


def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    net = EBGAN(args)
    net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EBGAN')

    # Optimization
    parser.add_argument('--epoch', default=20, type=int, help='epoch size')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--PT_ratio', default=0.1, type=float, help='cost weight of pulling-away term')
    parser.add_argument('--D_lr', default=1e-3, type=float, help='learning rate for the Discriminator')
    parser.add_argument('--G_lr', default=2e-3, type=float, help='learning rate for the Generator')
    parser.add_argument('--m', default=20, type=float, help='margin m')

    # Network
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dimension of Autoencoder')
    parser.add_argument('--noise_dim', default=100, type=int, help='noise dimension of Generator')
    parser.add_argument('--load_ckpt', default=True, type=str2bool, help='load previous checkpoint')
    parser.add_argument('--ckpt_dir', default='checkpoint', type=str, help='checkpoint directory')

    # Dataset
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CelebA', type=str, help='CelebA, CIFAR10')
    parser.add_argument('--num_workers', default=1, type=int, help='num_workers')

    # Visualization
    parser.add_argument('--env_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--output_dir', default='output', type=str, help='image output directory')
    parser.add_argument('--visdom', default=True, type=str2bool, help='enable visdom')
    parser.add_argument('--port', default=8097, type=int, help='visdom port')
    parser.add_argument('--sample_num', default=100, type=int, help='the number of sample generation')

    # misc
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')

    args = parser.parse_args()

    main(args)
