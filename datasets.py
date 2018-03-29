"""datasets.py"""

import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder

__datasets__ = ['cifar10', 'celeba']


class UnknownDatasetError(Exception):
    def __str__(self):
        return "unknown datasets error"


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    if name.lower() == 'cifar10':
        root = os.path.join(dset_dir, 'CIFAR10')
        train_kwargs = {'root':root, 'train':True, 'transform':transform, 'download':True}
        dset = CIFAR10
    elif name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        train_kwargs = {'root':root, 'transform':transform}
        dset = ImageFolder
    else:
        root = os.path.join(dset_dir, name)
        train_kwargs = {'root':root, 'transform':transform}
        dset = ImageFolder

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = dict()
    data_loader['train'] = train_loader

    return data_loader


if __name__ == '__main__':
    import argparse
    #os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--dset_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    data_loader = return_data(args)
