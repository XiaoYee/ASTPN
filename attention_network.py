import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import math



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
        self.pad1   = nn.ZeroPad2d((4,4,4,4))
        self.pad2   = nn.ZeroPad2d((4,4,4,4))
        self.pad3   = nn.ZeroPad2d((4,4,4,4))
        self.pad4   = nn.ZeroPad2d((2,2,0,0))
        self.conv_1 = nn.Conv2d(5,  16, 5,stride=1, padding=1)
        self.conv_2 = nn.Conv2d(16, 32, 5,stride=1, padding=1)
        self.conv_3 = nn.Conv2d(32, 32, 5,stride=1, padding=1)

        # self.spp_level = spp_level
        self.spp_layer = SPPLayer(4)
        self.fc1 = nn.Linear(32*(64+16+4+1), 128)

        # batch_first=True,then the input is provided as (batch,sequence,feature)
        self.rnn = nn.RNN(128,128,batch_first=True,dropout=0.6)
        self.h_0 = Variable(torch.randn(1, 1, 128).cuda())


    def forward(self, x):

        # print x.size()   #(16L, 5L, 56L, 40L)
        x = self.pad1(x)
        x = F.max_pool2d(F.tanh(self.conv_1(x)), 2, stride=2)
        x = self.pad2(x)
        x = F.max_pool2d(F.tanh(self.conv_2(x)), 2, stride=2)
        x = self.pad3(x)
        x = F.tanh(self.conv_3(x))
        # print x.size()   #(16L, 32L, 24L, 20L)
        # padding for SPP,for 20%8 !=0 
        x = self.pad4(x)
    	x = self.spp_layer(x)
    	x = F.dropout(x,0.6)   
    	x = self.fc1(x)
        # print x.size()
        x = x.view(1,-1,128)
        print x
    	x, hn = self.rnn(x, self.h_0)
    	return x



