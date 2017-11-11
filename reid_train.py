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

loss_diatance = nn.HingeEmbeddingLoss(3,size_average=False)
loss_identity = nn.CrossEntropyLoss()

#choose trainIDs split=0.5
IDs = os.listdir(osp.join(person_sequence,"cam1"))
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

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_log = []
for ep in range(nEpochs):
    # random every epoch
    order = torch.randperm(nTrainPersons)
    for i in range(nTrainPersons*2):
        # print i
        if (i%2 == 0): 
        # load data from same identity
            netInputA, netInputB, label_same = same_pair(trainID[order[i/2]],sampleSeqLength)	
            labelA = np.zeros(16, dtype=np.uint8)
            labelB = np.zeros(16, dtype=np.uint8)
            for m in range(16):
                labelA[m] = order[i/2]
                labelB[m] = order[i/2]
            labelA = torch.from_numpy(labelA)
            labelB = torch.from_numpy(labelB)
        else:
	        # load data from different identity random
	        netInputA, netInputB, labelA, labelB ,label_same = different_pair(trainID,sampleSeqLength)

        netInputA = Variable(torch.from_numpy(netInputA).float()).cuda()
        netInputB = Variable(torch.from_numpy(netInputB).float()).cuda()

        optimizer.zero_grad()

		# v_p,v_g,identity_p,identity_g = model(netInputA,netInputB)
        distance,identity_p,identity_g = model(netInputA,netInputB)

        label_same = torch.FloatTensor([label_same])
        label_same = Variable(label_same).cuda()

        label_identity1 = Variable(labelA.cuda().long())
        label_identity2 = Variable(labelB.cuda().long())
        # label_identity1 = Variable(label_identity1).cuda()

        # label_identity2 = torch.LongTensor([labelB])
        # label_identity2 = Variable(label_identity2).cuda()

        loss_pair = loss_diatance(distance,label_same)

        #two loss need to be fixed
        loss_identity1 = loss_identity(identity_p, label_identity1[0])
        loss_identity2 = loss_identity(identity_g, label_identity2[0])

        loss = loss_pair+loss_identity1+loss_identity2
        # loss = loss_pair

        # print loss
        # Euclidean_distance = (torch.mean(torch.pow((v_p-v_g),2))*(v_p.size()[0])).data.cpu()
        # zero = torch.FloatTensor([0])
        # label_same = Variable(label_same)
        # loss = label_same*Euclidean_distance+(1-label_same)*torch.max(zero, 3-Euclidean_distance)
        # loss = label_same*Euclidean_distance+(1-label_same)*torch.clamp((3-Euclidean_distance),min=0)
        # loss need to update

        if i%10 == 0:
            loss_log.append(loss.data[0])
            print('\nepoch: {} - batch: {}/{}'.format(ep, i, len(trainID)*2))
            print('loss: ', loss.data[0])

        # loss = nn.Parameter(loss)
        loss.backward() 
        optimizer.step()

# print loss_log
        
torch.save(model.state_dict(), './siamese.pth')

import matplotlib.pyplot as plt
plt.plot(loss_log)
plt.title('ASTPN Loss')
plt.show()























