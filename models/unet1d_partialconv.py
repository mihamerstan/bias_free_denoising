import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import register_model
from models import PartialConv1d
import math

### 1D PartialConv U-NET
@register_model("unet1d")
class UNet(nn.Module):
	"""UNet as defined in https://arxiv.org/abs/1805.07709"""
	def __init__(self, bias, residual_connection = False):
		super(UNet, self).__init__()

		self.conv1 = PartialConv1d.PartialConv1d(1,32,5,padding = 2, bias = bias, multi_channel=False)
		self.conv2 = PartialConv1d.PartialConv1d(32,32,3,padding = 1, bias = bias, multi_channel=False)
		self.conv3 = PartialConv1d.PartialConv1d(32,64,3,stride=2, padding = 1, bias = bias, multi_channel=False)
		self.conv4 = PartialConv1d.PartialConv1d(64,64,3,padding = 1, bias=bias, multi_channel=False)
		self.conv5 = PartialConv1d.PartialConv1d(64,64,3,dilation=2, padding = 2, bias = bias, multi_channel=False)
		self.conv6 = PartialConv1d.PartialConv1d(64,64,3,dilation = 4,padding = 4, bias = bias, multi_channel=False)
		self.conv7 = nn.ConvTranspose1d(64,64, 4,stride = 2, padding = 1, bias = bias)
		self.conv8 = PartialConv1d.PartialConv1d(96,32,3,padding=1, bias = bias, multi_channel=False)
		self.conv9 = PartialConv1d.PartialConv1d(32,1,5,padding = 2, bias = False, multi_channel=False)

		self.residual_connection = residual_connection;
		
	@staticmethod
	def add_args(parser):
		"""Add model-specific arguments to the parser."""
		# TO DO: add image_channels, other arguments
		parser.add_argument("--bias", action='store_true', help="use residual bias")
		parser.add_argument("--residual", action='store_true', help="use residual connection")

	@classmethod
	def build_model(cls, args):
		return cls(args.bias, args.residual)

	def forward(self, x, mask_in = None):
		# Pad both x and the mask in the last dimension
		# pad_right = x.shape[-2]%2
		pad_bottom = x.shape[-1]%2
		x = F.pad(x,[0,pad_bottom])
		mask_in = F.pad(mask_in,[0,pad_bottom])
		# Layer by layer. ReLu's can't take the mask as input
		prelu, mask = self.conv1(x, mask_in)
		out = F.relu(prelu)
		prelu, mask_saved = self.conv2(out, mask)
		out_saved = F.relu(prelu)
		prelu, mask = self.conv3(out_saved, mask_saved)
		out = F.relu(prelu)
		prelu, mask = self.conv4(out, mask)
		out = F.relu(prelu)
		prelu, mask = self.conv5(out, mask)
		out = F.relu(prelu)
		prelu, mask = self.conv6(out, mask)
		out = F.relu(prelu)
		out = F.relu(self.conv7(out)) # THE TRANSPOSE
		# Add the connection from layer 2
		out = torch.cat([out,out_saved],dim = 1)
		# Expand the mask [32,1,25] back to [32,1,50]
		mask = mask.repeat_interleave(2,dim=2)		

		prelu, mask = self.conv8(out,mask)
		out = F.relu(prelu)
		prelu, mask = self.conv9(out,mask)
		out = F.relu(prelu)

		if self.residual_connection:
			out = x - out;

		if pad_bottom:
			out = out[:, :, :-1]

		return out
