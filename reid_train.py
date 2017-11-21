import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from attention_network import AttentionNet
from compute_CMC import computeCMC
import random
import cv2
from dataset import same_pair,different_pair 

    
nEpochs = 600
learning_rate = 0.001
sampleSeqLength = 16

this_dir = osp.dirname(__file__)
person_sequence = osp.join(this_dir, "..", "data", "i-LIDS-VID", "sequences")
optical_sequence = osp.join(this_dir, "..", "data", "i-LIDS-VID-OF-HVP", "sequences")

model = AttentionNet()
model = model.cuda()

# set for optimizer step
# steps = [90000,500000]
# scales = [0.1,0.1]
# processed_batches = 0

loss_diatance = nn.HingeEmbeddingLoss(3,size_average=False)
loss_identity = nn.CrossEntropyLoss()

#choose trainIDs split=0.5
IDs = os.listdir(osp.join(person_sequence,"cam1"))
# print IDs
# print len(IDs) 300
trainID = []
testID  = []
for i in range(300):
	if i%2 == 0:
		trainID.append(IDs[i])
	else:
		testID.append(IDs[i])

nTrainPersons = len(trainID)
torch.manual_seed(1)

optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

def adjust_learning_rate(optimizer, batch):

    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

loss_log = []

for ep in range(nEpochs):
    # random every epoch
    model.train()
    loss_add = 0
    order = torch.randperm(nTrainPersons)
    for i in range(nTrainPersons*2):

        # lr = adjust_learning_rate(optimizer, processed_batches)
        # processed_batches = processed_batches + 1

        if (i%2 == 0): 
        # load data from same identity
            netInputA, netInputB, label_same = same_pair(trainID[order[i/2]],sampleSeqLength)	
            labelA = order[i/2]
            labelB = order[i/2]

        else:
	        # load data from different identity random
	        netInputA, netInputB, labelA, labelB ,label_same = different_pair(trainID,sampleSeqLength)

        netInputA = Variable(torch.from_numpy(netInputA.copy()).float()).cuda()
        netInputB = Variable(torch.from_numpy(netInputB.copy()).float()).cuda()

        # optimizer.zero_grad()
		# v_p,v_g,identity_p,identity_g = model(netInputA,netInputB)
        distance,identity_p,identity_g,v_p,v_g = model(netInputA,netInputB)


        label_same = torch.FloatTensor([label_same])
        label_same = Variable(label_same).cuda()


        label_identity1 = (Variable(torch.LongTensor([labelA]))).cuda()
        label_identity2 = (Variable(torch.LongTensor([labelB]))).cuda()

        loss_pair = loss_diatance(distance,label_same)

        #two loss need to be fixed
        loss_identity1 = loss_identity(identity_p, label_identity1)
        loss_identity2 = loss_identity(identity_g, label_identity2)

        loss = loss_pair+loss_identity1+loss_identity2

        loss_add = loss_add + loss.data[0]

        #### clip gradient parameters to train RNN #####

        # nn.utils.clip_grad_norm(model.parameters(), 5)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-5, 5)
        # torch.nn.utils.clip_grad_norm(model.parameters(),5)
        ##############################################
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

    if ep%1 == 0:
        # print('epoch %d, lr %f, loss %f ' % (ep, lr, loss_add))
        print('epoch %d, loss %f ' % (ep , loss_add))
        loss_log.append(loss_add)

    if (ep+1)%500 == 0:

        model.eval()
        cmc = computeCMC(testID, model)
        print cmc

    if (ep+1)%600 == 0:

        model.eval()
        cmc = computeCMC(testID, model)
        print cmc

        
torch.save(model.state_dict(), './siamese.pth')

import matplotlib.pyplot as plt
plt.plot(loss_log)
plt.title('ASTPN Loss')
plt.show()























