import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models import register_model
from models import BFBatchNorm2d, PartialConv1d
import math

@register_model("dncnn1d_partialconv")
class DnCNNPartialConv(nn.Module):
	def __init__(self, depth=20, n_channels=64, image_channels=2, use_bnorm=True, kernel_size=3):
		super(DnCNNPartialConv, self).__init__()
		kernel_size = 3
		padding = 1

		self.use_bnorm = use_bnorm;
		self.depth = depth;

		# self.first_layer = nn.Conv1d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False)
		self.first_layer = PartialConv1d.PartialConv1d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False, multi_channel=False)
		self.hidden_layer_list = [None] * (self.depth - 2);
		if self.use_bnorm:
			self.bn_layer_list = [None] * (self.depth -2 );
		
		for i in range(self.depth-2):
			# self.hidden_layer_list[i] = nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False);
			self.hidden_layer_list[i] = PartialConv1d.PartialConv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False, multi_channel=False)

			if self.use_bnorm:
				self.bn_layer_list[i] = nn.BatchNorm1d(n_channels, eps=0.0001, momentum = 0.95)
		
		self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list);
		if self.use_bnorm:
			self.bn_layer_list = nn.ModuleList(self.bn_layer_list);
		# self.last_layer = nn.Conv1d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False)
		self.last_layer = PartialConv1d.PartialConv1d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False, multi_channel=False)
		self._initialize_weights()


	@staticmethod
	def add_args(parser):
		"""Add model-specific arguments to the parser."""
		# TODO Add multi_channel
		parser.add_argument("--in-channels", type=int, default=2, help="number of image-channels")
		parser.add_argument("--hidden-size", type=int, default=64, help="hidden dimension (n_channels)")
		parser.add_argument("--num-layers", default=20, type=int, help="number of layers")
		parser.add_argument("--batchnorm", action='store_true', help="use batchnorm layers")
		parser.add_argument("--bias", action='store_true', help="use residual bias")

	@classmethod
	def build_model(cls, args):
		return cls(image_channels = args.in_channels, n_channels = args.hidden_size, depth = args.num_layers, use_bnorm=args.batchnorm)

	def forward(self, x, mask_in=None):
		
		out,mask = self.first_layer(x,mask_in);
		out = F.relu(out);
		for i in range(self.depth-2):
			out,mask = self.hidden_layer_list[i](out,mask);

			if self.use_bnorm:
				out = self.bn_layer_list[i](out);
			out = F.relu(out)

		out,mask = self.last_layer(out,mask);
		
		return out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv1d): # Does this still work or does it need to be PartialConv1d?
				init.orthogonal_(m.weight)
				# print('init weight')
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm1d):
				init.constant_(m.weight, 1)
				init.constant_(m.bias, 0)