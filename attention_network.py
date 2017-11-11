import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math


class MetrixMultiply(nn.Module):
    def __init__(self):
        super(MetrixMultiply, self).__init__()
        self.weight = nn.Parameter(torch.zeros(128,128))

    # reference:https://discuss.pytorch.org/t/how-to-define-a-new-layer-with-autograd/351
    def forward(self, x):
        return x.mm(self.weight)


class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    # need to fit arbitrary input size...reference:http://www.erogol.com/spp-network-pytorch/
    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


class AttentionNet(nn.Module):

    def __init__(self):
        super(AttentionNet, self).__init__()
        # Siamese Architecture
        self.cnn = nn.Sequential(
            nn.ZeroPad2d((4,4,4,4)),
            nn.Conv2d(5, 16, 5,stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2,stride=2),
            nn.ZeroPad2d((4,4,4,4)),
            nn.Conv2d(16, 32, 5,stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),
            nn.ZeroPad2d((4,4,4,4)),
            nn.Conv2d(32, 32, 5,stride=1, padding=1),
            nn.Tanh(),
            nn.ZeroPad2d((2,2,0,0)),
            SPPLayer(4),
            nn.Dropout(0.6),
            nn.Linear(32*(64+16+4+1), 128)
            )
        # batch_first=True,then the input is provided as (batch,sequence,feature)
        self.rnn = nn.RNN(128,128,batch_first=True,dropout=0.6)
        self.h_0 = Variable(torch.randn(1, 1, 128).cuda())
        self.Metrix = MetrixMultiply()
        self.fc = nn.Linear(128, 150)


    def forward(self,x,y):
        #input two data
        x = self.cnn(x)
        y = self.cnn(y)
        x = x.view(1,-1,128)
        y = y.view(1,-1,128)
    	x, hn = self.rnn(x, self.h_0)
        y, hn = self.rnn(y, self.h_0)
        P = x.squeeze(0)
        G = y.squeeze(0)
        #attention begin
        A = self.Metrix(P)
        A = F.tanh(A.mm(torch.t(G)))
        t_p,indx = torch.max(A,0)
        t_g,indx = torch.max(A,1)
        # notice don't view as (16,1),otherwise soft will be all 1
        t_p = t_p.view(1,16)
        t_g = t_g.view(1,16)
        soft = nn.Softmax()
        a_p = soft(t_p)
        a_g = soft(t_g)
        # print a_p.size()
        v_p = a_p.mm(P)
        v_g = a_g.mm(G)
        f_p = self.fc(v_p)
        f_g = self.fc(v_g)
        # logsoft = nn.LogSoftmax()
        # loss_p = logsoft(v_p)
        # loss_g = logsoft(v_g)
        # use softmax and cross-entropy loss as paper
        identity_p = soft(f_p)
        identity_g = soft(f_g)

        pdist = nn.PairwiseDistance(p=2)
        distance = pdist(v_p, v_g)

    	return distance,identity_p,identity_g






