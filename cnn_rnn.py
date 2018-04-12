# -*- coding: utf-8 -*-
import torch.nn as nn
from torch.autograd import Variable

class cnn_rnn(nn.Module):
	def __init__(self,filters):
		super(cnn_rnn, self).__init__()
		# parameters for cnn
		filtsize = [5,5,5,1]
		# poolsize = [2,2,2]
		# stepSize = [2,2,2]
		padDim = 4
		ninputChannels = 5
		self.tanh = nn.Tanh()
		# nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
		self.maxpool_s2 = nn.MaxPool2d(2,2)

		self.conv1 = nn.Sequential(
			# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
			nn.Conv2d(ninputChannels, filters[0], filtsize[0], 1, padDim),
			self.tanh,
			self.maxpool_s2
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(filters[0], filters[1], filtsize[1], 1, padDim),
			self.tanh,
			self.maxpool_s2
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(filters[1], filters[2], filtsize[2], 1, padDim),
			self.tanh,
			self.maxpool_s2
		)


	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				stdv = 1./math.sqrt(m.kernel_size[0] * m.kernel_size[1] * m.in_channels)
				nn.init.uniform(m.weight.data, -stdv, stdv)
				if m.bias is not None:
					nn.init.uniform(m.bias.data, -stdv, stdv)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				stdv = 1./math.sqrt(m.weight.size(1))
				nn.init.uniform(m.weight.data, -stdv, stdv)
				if m.bias is not None:
					nn.init.uniform(m.bias.data, -stdv, stdv)

	def forward(self, input_x):
		o_conv1	= self.conv1(input_x)
		o_conv2	= self.conv2(o_conv1)
		o_conv3	= self.conv3(o_conv2)

		return o_conv3

