import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable



class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    # need to update......reference:http://www.erogol.com/spp-network-pytorch/  
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

    def __init__(self,spp_level=4):
        super(AttentionNet, self).__init__()
        # Siamese Architecture
        self.cnn = nn.Sequential(
        	nn.Conv2d(5,16,5,stride=1, padding=4),
        	nn.Tanh(),
        	nn.MaxPool2d(2,stride=2),
        	nn.Conv2d(5,32,5,stride=1, padding=4),
        	nn.Tanh(),
			nn.MaxPool2d(2,stride=2),
			nn.Conv2d(5,32,5,stride=1, padding=4),
			nn.Tanh()
        	)
        self.spp_level = spp_level
        self.spp_layer = SPPLayer(spp_level)
        self.fc1 = nn.Linear(32*(64+16+4+1), 128)
        self.rnn = nn.RNN(128,128,batch_first=True,dropout=0.6)
        self.h_0 = Variable(torch.randn(1, 1, 128))


    def forward(self, x):

    	x = self.cnn(x)
    	x = self.spp_layer(x)
    	x = x.view(-1, 32*(64+16+4+1)) 
    	x = F.dropout(x,0.6)   
    	x = self.fc1(x)
    	x, hn = rnn(x, self.h_0)
    	return x



