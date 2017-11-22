import os
import os.path as osp
import random
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# sampleSeqLength = 16
this_dir = osp.dirname(__file__)
person_sequence = osp.join(this_dir, "..", "data", "i-LIDS-VID", "sequences")
optical_sequence = osp.join(this_dir, "..", "data", "i-LIDS-VID-OF-HVP", "sequences")

im_mean = [124, 117, 104]
im_std = [0.229, 0.224, 0.225]

def same_pair(batch_number, sampleSeqLength, is_train=True):

    image_cam1 = os.listdir(osp.join(person_sequence,"cam1",str(batch_number)))
    optical_cam1 = os.listdir(osp.join(optical_sequence,"cam1",str(batch_number)))
    image_cam2 = os.listdir(osp.join(person_sequence,"cam2",str(batch_number)))
    optical_cam2 = os.listdir(osp.join(optical_sequence,"cam2",str(batch_number)))
    # print batch_number
    image_cam1.sort()
    image_cam2.sort()
    optical_cam1.sort()
    optical_cam2.sort()
    len_cam1 = len(image_cam1)
    len_cam2 = len(image_cam2)
    actualSampleSeqLen = sampleSeqLength
    startA = int(random.random()* ((len_cam1 - actualSampleSeqLen) + 1))   
    startB = int(random.random()* ((len_cam2 - actualSampleSeqLen) + 1)) 
    # print startA,startB
    netInputA = np.zeros((64, 48, 5, actualSampleSeqLen), dtype=np.float32)
    netInputB = np.zeros((64, 48, 5, actualSampleSeqLen), dtype=np.float32)
    #########for debug not using optical#############
    # netInputA = np.zeros((64, 48, 3, actualSampleSeqLen), dtype=np.float32)
    # netInputB = np.zeros((64, 48, 3, actualSampleSeqLen), dtype=np.float32)

    ######for Data augmentation############################################
    crpxA = int(random.random()*8)+1
    crpyA = int(random.random()*8)+1
    crpxB = int(random.random()*8)+1
    crpyB = int(random.random()*8)+1
    flipA = int(random.random()*2)+1
    flipB = int(random.random()*2)+1
    #######################################################################

    for m in range(actualSampleSeqLen):
    	img_file = os.path.join(person_sequence,"cam1",str(batch_number),image_cam1[startA+m])
    	img = cv2.imread(img_file)
        # BGR TO YUV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    	img = cv2.resize(img,(48,64))
        m0  = np.mean(img[:,:,0]) 
        m1  = np.mean(img[:,:,1])
        m2  = np.mean(img[:,:,2])
        v0  = np.sqrt(np.var(img[:,:,0]))
        v1  = np.sqrt(np.var(img[:,:,1])) 
        v2  = np.sqrt(np.var(img[:,:,2])) 
    	netInputA[:, :, 0, m] = (img[:,:,0]-m0)/np.sqrt(v0)
    	netInputA[:, :, 1, m] = (img[:,:,1]-m1)/np.sqrt(v1)
    	netInputA[:, :, 2, m] = (img[:,:,2]-m2)/np.sqrt(v2)
    	optical_file = os.path.join(optical_sequence,"cam1",str(batch_number),optical_cam1[startA+m])
        # print optical_file
    	optical = cv2.imread(optical_file)
    	optical = cv2.resize(optical,(48,64))
        m3  = np.mean(optical[:,:,1]) 
        m4  = np.mean(optical[:,:,2])
        v3  = np.sqrt(np.var(optical[:,:,1])) 
        v4  = np.sqrt(np.var(optical[:,:,2])) 
    	netInputA[:, :, 3, m] = (optical[:,:,1]-m3)/np.sqrt(v3)
    	netInputA[:, :, 4, m] = (optical[:,:,2]-m4)/np.sqrt(v4)

    for m in range(actualSampleSeqLen):
    	img_file = os.path.join(person_sequence,"cam2",str(batch_number),image_cam2[startB+m])
    	img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    	img = cv2.resize(img,(48,64))
    	m0  = np.mean(img[:,:,0]) 
        m1  = np.mean(img[:,:,1])
        m2  = np.mean(img[:,:,2])
        v0  = np.sqrt(np.var(img[:,:,0]))
        v1  = np.sqrt(np.var(img[:,:,1])) 
        v2  = np.sqrt(np.var(img[:,:,2])) 
        netInputB[:, :, 0, m] = (img[:,:,0]-m0)/np.sqrt(v0)
        netInputB[:, :, 1, m] = (img[:,:,1]-m1)/np.sqrt(v1)
        netInputB[:, :, 2, m] = (img[:,:,2]-m2)/np.sqrt(v2)
    	optical_file = os.path.join(optical_sequence,"cam2",str(batch_number),optical_cam2[startB+m])
    	optical = cv2.imread(optical_file)
    	optical = cv2.resize(optical,(48,64))
        m3  = np.mean(optical[:,:,1]) 
        m4  = np.mean(optical[:,:,2])
        v3  = np.sqrt(np.var(optical[:,:,1])) 
        v4  = np.sqrt(np.var(optical[:,:,2])) 
        netInputB[:, :, 3, m] = (optical[:,:,1]-m3)/np.sqrt(v3)
        netInputB[:, :, 4, m] = (optical[:,:,2]-m4)/np.sqrt(v4)

    netInputA = np.transpose(netInputA, (3,2,0,1))
    netInputB = np.transpose(netInputB, (3,2,0,1))
    # print netInputA
    netInputA = doDataAug(netInputA,crpxA,crpyA,flipA)
    # print netInputA
    netInputB = doDataAug(netInputB,crpxB,crpyB,flipB)

    label_same = 1

    return netInputA,netInputB,label_same


def different_pair(trainID, sampleSeqLength, is_train=True):
	
    train_probe_num ,train_gallery_num = random.sample(range(150), 2)
    train_probe = trainID[train_probe_num]
    train_gallery = trainID[train_gallery_num]
    image_cam1 = os.listdir(osp.join(person_sequence,"cam1",str(train_probe)))
    optical_cam1 = os.listdir(osp.join(optical_sequence,"cam1",str(train_probe)))
    image_cam2 = os.listdir(osp.join(person_sequence,"cam2",str(train_gallery)))
    optical_cam2 = os.listdir(osp.join(optical_sequence,"cam2",str(train_gallery)))
    image_cam1.sort()
    image_cam2.sort()
    optical_cam1.sort()
    optical_cam2.sort()
    len_cam1 = len(image_cam1)
    len_cam2 = len(image_cam2)
    actualSampleSeqLen = sampleSeqLength
    startA = int(random.random()* ((len_cam1 - actualSampleSeqLen) + 1))    
    startB = int(random.random()* ((len_cam2 - actualSampleSeqLen) + 1)) 

    netInputA = np.zeros((64, 48, 5, actualSampleSeqLen), dtype=np.float32)
    netInputB = np.zeros((64, 48, 5, actualSampleSeqLen), dtype=np.float32)
    # netInputA = np.zeros((64, 48, 3, actualSampleSeqLen), dtype=np.float32)
    # netInputB = np.zeros((64, 48, 3, actualSampleSeqLen), dtype=np.float32)

    crpxA = int(random.random()*8)+1
    crpyA = int(random.random()*8)+1
    crpxB = int(random.random()*8)+1
    crpyB = int(random.random()*8)+1
    flipA = int(random.random()*2)+1
    flipB = int(random.random()*2)+1

    for m in range(actualSampleSeqLen):
    	img_file = os.path.join(person_sequence,"cam1",str(train_probe),image_cam1[startA+m])
    	img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    	img = cv2.resize(img,(48,64))
    	m0  = np.mean(img[:,:,0]) 
        m1  = np.mean(img[:,:,1])
        m2  = np.mean(img[:,:,2])
        v0  = np.sqrt(np.var(img[:,:,0]))
        v1  = np.sqrt(np.var(img[:,:,1])) 
        v2  = np.sqrt(np.var(img[:,:,2])) 
        netInputA[:, :, 0, m] = (img[:,:,0]-m0)/np.sqrt(v0)
        netInputA[:, :, 1, m] = (img[:,:,1]-m1)/np.sqrt(v1)
        netInputA[:, :, 2, m] = (img[:,:,2]-m2)/np.sqrt(v2)
    	optical_file = os.path.join(optical_sequence,"cam1",str(train_probe),optical_cam1[startA+m])
    	optical = cv2.imread(optical_file)
    	optical = cv2.resize(optical,(48,64))
        m3  = np.mean(optical[:,:,1]) 
        m4  = np.mean(optical[:,:,2])
        v3  = np.sqrt(np.var(optical[:,:,1])) 
        v4  = np.sqrt(np.var(optical[:,:,2])) 
        netInputA[:, :, 3, m] = (optical[:,:,1]-m3)/np.sqrt(v3)
        netInputA[:, :, 4, m] = (optical[:,:,2]-m4)/np.sqrt(v4)


    for m in range(actualSampleSeqLen):
    	img_file = os.path.join(person_sequence,"cam2",str(train_gallery),image_cam2[startB+m])
    	img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    	img = cv2.resize(img,(48,64))
        m0  = np.mean(img[:,:,0]) 
        m1  = np.mean(img[:,:,1])
        m2  = np.mean(img[:,:,2])
        v0  = np.sqrt(np.var(img[:,:,0]))
        v1  = np.sqrt(np.var(img[:,:,1])) 
        v2  = np.sqrt(np.var(img[:,:,2])) 
        netInputB[:, :, 0, m] = (img[:,:,0]-m0)/np.sqrt(v0)
        netInputB[:, :, 1, m] = (img[:,:,1]-m1)/np.sqrt(v1)
        netInputB[:, :, 2, m] = (img[:,:,2]-m2)/np.sqrt(v2)
    	optical_file = os.path.join(optical_sequence,"cam2",str(train_gallery),optical_cam2[startB+m])
    	optical = cv2.imread(optical_file)
    	optical = cv2.resize(optical,(48,64))
    	m3  = np.mean(optical[:,:,1]) 
        m4  = np.mean(optical[:,:,2])
        v3  = np.sqrt(np.var(optical[:,:,1])) 
        v4  = np.sqrt(np.var(optical[:,:,2])) 
        netInputB[:, :, 3, m] = (optical[:,:,1]-m3)/np.sqrt(v3)
        netInputB[:, :, 4, m] = (optical[:,:,2]-m4)/np.sqrt(v4)


    netInputA = np.transpose(netInputA, (3,2,0,1))
    netInputB = np.transpose(netInputB, (3,2,0,1))

    netInputA = doDataAug(netInputA,crpxA,crpyA,flipA)
    netInputB = doDataAug(netInputB,crpxB,crpyB,flipB)


    labelA = train_probe_num
    labelB = train_gallery_num


    label_same = -1
    return netInputA,netInputB,labelA,labelB,label_same


def doDataAug(netInput,crpx,crpy,flip):

    netInput = netInput[:,:,crpy:56+crpy,crpx:40+crpx]
    if flip == 1:
        netInput = netInput[:,:,:,::-1]
    else:
        netInput = netInput
    return netInput


if __name__ == '__main__':
    
    IDs = os.listdir(osp.join(person_sequence,"cam1"))
    trainID = []
    testID  = []
    for i in range(300):
	# print IDs[i]
	if i%2 == 0:
		trainID.append(IDs[i])
	else:
		testID.append(IDs[i])
    # print trainID
    for batch_n in range(len(trainID)*2):
        if (batch_n%2 == 0): 
            # load data from same identity
            netInputA, netInputB,label_same = same_pair(trainID[batch_n/2],16) 
        else:
            # load data from different identity random
            netInputA, netInputB,labelA,labelB,label_same = different_pair(trainID,16)
    # netInputA,netInputB = different_pair(trainID,16)
    


