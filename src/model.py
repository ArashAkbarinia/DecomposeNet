"""
VQ-VAE Model
"""

import numpy as np
import logging

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

import nearest_embed


class ResBlock(nn.Module):
    def __init__(self, in_chns, out_chns, mid_chns=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_chns is None:
            mid_chns = out_chns

        self.in_chns = in_chns
        self.out_chns = out_chns
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_chns, mid_chns, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_chns, out_chns, kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_chns))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        if self.in_chns == self.out_chns:
            return x + self.convs(x)
        else:
            return self.convs(x)


class HueLoss(torch.nn.Module):
    def forward(self, recon_x, x):
        ret = recon_x - x
        ret[ret > 1] -= 2
        ret[ret < -1] += 2
        ret = ret ** 2
        return torch.mean(ret)


class VQ_CVAE(nn.Module):
    def __init__(self, d, k=10, kl=None, vq_coef=1, commit_coef=0.5,
                 in_chns=3, colour_space='rgb', out_chns=3):
        super(VQ_CVAE, self).__init__()

        if out_chns is None:
            out_chns = in_chns
        self.out_chns = out_chns

        self.d = d
        self.k = k
        if kl is None:
            kl = d
        self.kl = kl
        self.emb = nearest_embed.NearestEmbed(k, kl)

        self.colour_space = colour_space
        self.hue_loss = HueLoss()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chns, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, kl, bn=True),
            nn.BatchNorm2d(kl),
        )
        self.decoder = nn.Sequential(
            ResBlock(kl, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, out_chns, kernel_size=4, stride=2, padding=1)
        )
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin

    def sample(self, size):
        if self.cuda():
            sample = torch.tensor(
                torch.randn(size, self.kl, self.f, self.f), requires_grad=False
            ).cuda()
        else:
            sample = torch.tensor(
                torch.randn(size, self.kl, self.f, self.f), requires_grad=False
            )
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.kl, self.f, self.f)).cpu()

    def sample_inds(self, inds):
        assert len(inds.shape) == 2
        rows = inds.shape[0]
        cols = inds.shape[1]
        inds = inds.reshape(rows * cols)
        weights = self.emb.weight.detach().cpu().numpy()
        sample = np.zeros((self.kl, rows, cols))
        sample = sample.reshape(self.kl, rows * cols)
        for i in range(self.k):
            which_inds = inds == i
            sample[:, which_inds] = np.broadcast_to(
                weights[:, i], (which_inds.sum(), self.kl)
            ).T
        sample = sample.reshape(self.kl, rows, cols)
        emb = torch.tensor(sample, dtype=torch.float32).unsqueeze(dim=0)
        return self.decode(emb).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        if self.colour_space == 'hsv':
            self.mse = F.mse_loss(recon_x[:, 1:], x[:, 1:])
            self.mse += self.hue_loss(recon_x[:, 0], x[:, 0])
        else:
            self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e) ** 2, 2, 1))

        return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss,
                'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):
        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)
