# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from cnn_rnn import cnn_rnn

class MetrixMultiply(nn.Module):
	def __init__(self):
		super(MetrixMultiply, self).__init__()
		self.weight = nn.Parameter(torch.Tensor(128, 128).uniform_(-1/128, 1/128),requires_grad = True)

	def forward(self, x):
		return x.mm(self.weight)


class ASTPN(nn.Module):

	def __init__(self):
		super(ASTPN, self).__init__()
		
		filters = [16,32,32]

		self.cnn1 = cnn_rnn(filters)
		# nFullyConnected = filters[-2]*14*8 #32*10*8, 32*8*6
		nFullyConnected = filters[-2]*10*8 #32*10*8, 32*8*6

		
		self.cnn2rnn = nn.Sequential(
			nn.Dropout(0.6),
			nn.Linear(nFullyConnected, 128)
		)

		# parameters for rnn
		self.nhid = 128
		self.nlayers = 1
		self.bsz = 1

		self.h2h = nn.Linear(self.nhid, self.nhid)
		self.i2h = nn.Linear(self.nhid, self.nhid)
		
		self.clsLayer = nn.Linear(self.nhid, 150)
		self._initialize_weights()

		self.Metrix = MetrixMultiply()

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

	def forward_cnn(self, x):
		# print 'x',x.size()
		cnnOutput = self.cnn1(x)
		output = cnnOutput.view(cnnOutput.size()[0], -1)
		output = self.cnn2rnn(output) #[16L, 128L]
		return output


	def forward_RNN(self, x, ox):
		x = x.view(1, -1)
		x = self.i2h(x)
		ox = self.h2h(ox)
		hx = F.dropout(F.tanh(x+ox), p=0.6, training=self.training, inplace=False)

		return hx


	def init_hidden(self):

		h_0 = Variable(torch.zeros(self.bsz, self.nhid).float()).cuda()
		return h_0

	def forward(self, input_x, input_y):
		cnn_x = self.forward_cnn(input_x)
		cnn_y = self.forward_cnn(input_y)
		# rnn forward
		x_Output = []
		x_sOutput = []
		y_Output = []
		y_sOutput = []

		ox = self.init_hidden()
		oy = self.init_hidden()
		for idx in range(cnn_x.size()[0]):
			ox = self.forward_RNN(cnn_x[idx],ox)
			x_Output.append(ox)
			
		for idx in range(cnn_y.size()[0]):
			oy = self.forward_RNN(cnn_y[idx],oy)
			y_Output.append(oy)

		x_Output = torch.cat(x_Output, 0)
		y_Output = torch.cat(y_Output, 0)
		

		####attention begin########
		A = self.Metrix(x_Output)
		A = F.tanh(A.mm(torch.t(y_Output)))
		t_p,indx = torch.max(A,1)
		t_g,indx = torch.max(A,0)
		t_p = t_p.view(1,-1)
		t_g = t_g.view(1,-1)
		a_p = nn.Softmax(dim=1)(t_p)
		a_g = nn.Softmax(dim=1)(t_g)
		# print a_p.size()
		x_Output = a_p.mm(x_Output)
		y_Output = a_g.mm(y_Output)
		#####attention end#######

		f_p = self.clsLayer(x_Output)
		f_g = self.clsLayer(y_Output)

		pdist = nn.PairwiseDistance(p=2)
		distance = pdist(x_Output, y_Output)


		return distance,f_p,f_g,x_Output, y_Output