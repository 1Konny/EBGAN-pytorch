"""solver.py"""

import time
from pathlib import Path

import visdom
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda
from model import Discriminator, Generator
from datasets import return_data


class EBGAN(object):
    def __init__(self, args):
        # misc
        self.args = args
        self.cuda = args.cuda and torch.cuda.is_available()
        self.seed = args.seed

        # Optimization
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.PT_ratio = args.PT_ratio
        self.D_lr = args.D_lr
        self.G_lr = args.G_lr
        self.m = args.m
        self.global_epoch = 0
        self.global_iter = 0

        # Network
        self.hidden_dim = args.hidden_dim
        self.noise_dim = args.noise_dim
        self.fixed_z = self.sample_z(args.sample_num)
        self.fixed_z = Variable(cuda(self.fixed_z, self.cuda))
        self.load_ckpt = args.load_ckpt
        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        self.model_init()

        # Dataset
        self.dataset = args.dataset
        self.data_loader = return_data(args)

        # Visualization
        self.env_name = args.env_name
        self.visdom = args.visdom
        self.port = args.port
        self.sample_num = args.sample_num

        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        self.visualization_init()

        self.lr_step_size = len(self.data_loader['train'].dataset)//self.batch_size*self.epoch//2

    def visualization_init(self):
        if self.visdom:
            self.viz_train_samples = visdom.Visdom(env=self.env_name+'/train_samples', port=self.port)
            self.viz_test_samples = visdom.Visdom(env=self.env_name+'/test_samples', port=self.port)

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def model_init(self):
        self.D = cuda(Discriminator(self.hidden_dim), self.cuda)
        self.G = cuda(Generator(self.noise_dim), self.cuda)

        self.D.weight_init(mean=0.0, std=0.02)
        self.G.weight_init(mean=0.0, std=0.02)

        self.D_optim = optim.Adam(self.D.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        self.G_optim = optim.Adam(self.G.parameters(), lr=self.G_lr, betas=(0.5, 0.999))

        self.D_optim_scheduler = lr_scheduler.StepLR(self.D_optim, step_size=1, gamma=0.5)
        self.G_optim_scheduler = lr_scheduler.StepLR(self.G_optim, step_size=1, gamma=0.5)

        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.load_ckpt:
            self.load_checkpoint()

    def train(self):
        criterion = F.mse_loss
        #criterion = F.l1_loss
        for e in range(self.epoch):
            self.global_epoch += 1
            elapsed = time.time()

            for idx, (images, labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1
                self.set_mode('train')

                # Discriminator Training
                x_real = Variable(cuda(images, self.cuda))
                D_real = self.D(x_real)[0]
                D_loss_real = criterion(D_real, x_real)

                z = self.sample_z()
                z = Variable(cuda(z, self.cuda))
                x_fake = self.G(z)
                D_fake = self.D(x_fake.detach())[0]
                D_loss_fake = criterion(D_fake, x_fake)

                #D_loss = D_loss_real + (self.m-D_loss_fake).clamp(min=0)
                D_loss = D_loss_real
                if D_loss_fake.data[0] < self.m:
                    D_loss += (self.m-D_loss_fake)

                self.D_optim.zero_grad()
                D_loss.backward()
                self.D_optim.step()

                # Generator Training
                z = self.sample_z()
                z = Variable(cuda(z, self.cuda))
                x_fake = self.G(z)
                D_fake, D_hidden = self.D(x_fake)
                G_loss_PT = self.repelling_regularizer(D_hidden, D_hidden)
                G_loss_fake = criterion(x_fake, D_fake)
                G_loss = G_loss_fake + self.PT_ratio*G_loss_PT

                self.G_optim.zero_grad()
                G_loss.backward()
                self.G_optim.step()

                if self.global_iter%200 == 0:
                    print()
                    print(self.global_iter)
                    print('D_loss_real:{:.3f} D_loss_fake:{:.3f}'.
                          format(D_loss_real.data[0], D_loss_fake.data[0]))
                    print('G_loss_fake:{:.3f} G_loss_PT:{:.3f}'.
                          format(G_loss_fake.data[0], G_loss_PT.data[0]))

                if self.global_iter%500 == 0 and self.visdom:
                    self.viz_train_samples.images(self.unscale(x_fake).cpu().data)
                    self.viz_train_samples.images(self.unscale(D_fake).cpu().data)
                    self.viz_train_samples.images(self.unscale(x_real).cpu().data)
                    self.viz_train_samples.images(self.unscale(D_real).cpu().data)

                if self.global_iter%1000 == 0 and self.visdom:
                    self.sample_img('fixed')
                    self.sample_img('random')
                    self.save_checkpoint()

                if self.global_iter%self.lr_step_size == 0:
                    self.scheduler_step()

            elapsed = (time.time()-elapsed)
            print()
            print('epoch {:d}, [{:.2f}s]'.format(e, elapsed))

        print("[*] Training Finished!")

    def repelling_regularizer(self, s1, s2):
        """Calculate Pulling-away Term(PT)."""

        n = s1.size(0)
        s1 = F.normalize(s1, p=2, dim=1)
        s2 = F.normalize(s2, p=2, dim=1)

        S1 = s1.unsqueeze(1).repeat(1, s2.size(0), 1)
        S2 = s2.unsqueeze(0).repeat(s1.size(0), 1, 1)

        f_PT = S1.mul(S2).sum(-1).pow(2)
        f_PT = torch.tril(f_PT, -1).sum().mul(2).div((n*(n-1)))

        #f_PT = (S1.mul(S2).sum(-1).pow(2).sum(-1)-1).sum(-1).div(n*(n-1))
        return f_PT

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.G.train()
            self.D.train()
        elif mode == 'eval':
            self.G.eval()
            self.D.eval()
        else:
            raise('mode error. It should be either train or eval')

    def scheduler_step(self):
        self.D_optim_scheduler.step()
        self.G_optim_scheduler.step()

    def unscale(self, tensor):
        return tensor.mul(0.5).add(0.5)

    def sample_z(self, batch_size=0, dim=0, dist='normal'):
        if batch_size == 0:
            batch_size = self.batch_size
        if dim == 0:
            dim = self.noise_dim

        if dist == 'normal':
            return torch.randn(batch_size, dim)
        elif dist == 'uniform':
            return torch.rand(batch_size, dim).mul(2).add(-1)
        else:
            return None

    def sample_img(self, _type='fixed', nrow=10):
        self.set_mode('eval')

        if _type == 'fixed':
            z = self.fixed_z
        elif _type == 'random':
            z = self.sample_z(self.sample_num)
            z = Variable(cuda(z, self.cuda))
        else:
            self.set_mode('train')
            return

        samples = self.unscale(self.G(z))
        samples = samples.data.cpu()

        filename = self.output_dir.joinpath(_type+':'+str(self.global_iter)+'.jpg')
        grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)
        save_image(grid, filename=filename)
        if self.visdom:
            self.viz_test_samples.image(grid, opts=dict(title=str(filename), nrow=nrow, factor=2))

        self.set_mode('train')

    def save_checkpoint(self, filename='ckpt.tar'):
        model_states = {'G':self.G.state_dict(),
                        'D':self.D.state_dict()}
        optim_states = {'G_optim':self.G_optim.state_dict(),
                        'D_optim':self.D_optim.state_dict()}
        states = {'args':self.args,
                  'iter':self.global_iter,
                  'epoch':self.global_epoch,
                  'fixed_z':self.fixed_z.data.cpu(),
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = self.ckpt_dir.joinpath(filename)
        torch.save(states, file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='ckpt.tar'):
        file_path = self.ckpt_dir.joinpath(filename)
        if file_path.is_file():
            checkpoint = torch.load(file_path.open('rb'))
            self.global_iter = checkpoint['iter']
            self.global_epoch = checkpoint['epoch']
            self.fixed_z = checkpoint['fixed_z']
            self.fixed_z = Variable(cuda(self.fixed_z, self.cuda))
            self.G.load_state_dict(checkpoint['model_states']['G'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.G_optim.load_state_dict(checkpoint['optim_states']['G_optim'])
            self.D_optim.load_state_dict(checkpoint['optim_states']['D_optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
