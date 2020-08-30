import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import register_model
import math

@register_model("gan_discriminator")
class Discriminator(nn.Module):
    def __init__(self, n_channels=64, image_channels=1, bias=False, kernel_size=4):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv1d(image_channels, n_channels, kernel_size, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv1d(n_channels, n_channels * 2, kernel_size, 2, 1, bias=False),
            nn.BatchNorm1d(n_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size, 2, 1, bias=False),
            nn.BatchNorm1d(n_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv1d(n_channels * 4, n_channels * 8, kernel_size, 2, 1, bias=False),
            nn.BatchNorm1d(n_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv1d(n_channels * 8, 1, kernel_size, 1, 0, bias=False),
            nn.Sigmoid()
        )
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # TODO Add return_mask and multi_channel
        parser.add_argument("--in_channels", type=int, default=1, help="number of image-channels")
        parser.add_argument("--hidden_size", type=int, default=64, help="hidden dimension")

        # parser.add_argument("--batchnorm", action='store_true', help="use batchnorm layers")
        # parser.add_argument("--bias", action='store_true', help="use residual bias")

    @classmethod
    def build_model(cls, args):
        return cls(image_channels = args.in_channels, n_channels = args.hidden_size)

    def forward(self, input):
        return self.main(input)

