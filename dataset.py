import os
import os.path as osp
import random
import numpy as np
import cv2

# sampleSeqLength = 16
this_dir = osp.dirname(__file__)
person_sequence = osp.join(this_dir, "data", "i-LIDS-VID", "sequences")
optical_sequence = osp.join(this_dir, "data", "i-LIDS-VID-OF-HVP", "sequences")



def same_pair(batch_number, sampleSeqLength, is_train=True):
	# print batch_number
    image_cam1 = os.listdir(osp.join(person_sequence,"cam1",str(batch_number)))
    optical_cam1 = os.listdir(osp.join(optical_sequence,"cam1",str(batch_number)))
    image_cam2 = os.listdir(osp.join(person_sequence,"cam2",str(batch_number)))
    optical_cam2 = os.listdir(osp.join(optical_sequence,"cam2",str(batch_number)))
    image_cam1.sort()
    image_cam2.sort()
    optical_cam1.sort()
    optical_cam2.sort()
    len_cam1 = len(image_cam1)
    len_cam2 = len(image_cam2)
    actualSampleSeqLen = sampleSeqLength
    startA = int(random.random()* ((len_cam1 - actualSampleSeqLen) + 1)) + 1    
    startB = int(random.random()* ((len_cam2 - actualSampleSeqLen) + 1)) + 1
    # print startA,startB
    netInputA = np.zeros((56, 40, 5, actualSampleSeqLen), dtype=np.float32)
    netInputB = np.zeros((56, 40, 5, actualSampleSeqLen), dtype=np.float32)
    for m in range(actualSampleSeqLen):
    	img_file = os.path.join(person_sequence,"cam1",str(batch_number),image_cam1[startA+m])
    	img = cv2.imread(img_file)
    	img = cv2.resize(img,(40,56))
    	netInputA[:, :, 0, m] = img[:,:,0]
    	netInputA[:, :, 1, m] = img[:,:,1]
    	netInputA[:, :, 2, m] = img[:,:,2]
    	optical_file = os.path.join(optical_sequence,"cam1",str(batch_number),optical_cam1[startA+m])
    	optical = cv2.imread(optical_file)
    	optical = cv2.resize(optical,(40,56))
    	netInputA[:, :, 3, m] = optical[:,:,0]
    	netInputA[:, :, 4, m] = optical[:,:,1]

    for m in range(actualSampleSeqLen):
    	img_file = os.path.join(person_sequence,"cam2",str(batch_number),image_cam2[startB+m])
    	img = cv2.imread(img_file)
    	img = cv2.resize(img,(40,56))
    	netInputB[:, :, 0, m] = img[:,:,0]
    	netInputB[:, :, 1, m] = img[:,:,1]
    	netInputB[:, :, 2, m] = img[:,:,2]
    	optical_file = os.path.join(optical_sequence,"cam2",str(batch_number),optical_cam2[startB+m])
    	optical = cv2.imread(optical_file)
    	optical = cv2.resize(optical,(40,56))
    	netInputB[:, :, 3, m] = optical[:,:,0]
    	netInputB[:, :, 4, m] = optical[:,:,1]
    netInputA = np.transpose(netInputA, (3,2,0,1))
    netInputA = np.transpose(netInputB, (3,2,0,1))
    return netInputA,netInputB


def different_pair(trainID, sampleSeqLength, is_train=True):
	
    # trainID_random = random.shuffle(trainID)
    train_probe ,train_gallery = random.sample(range(150), 2)
    print train_probe
    train_probe = trainID[train_probe]
    train_gallery = trainID[train_gallery]
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
    startA = int(random.random()* ((len_cam1 - actualSampleSeqLen) + 1)) + 1    
    startB = int(random.random()* ((len_cam2 - actualSampleSeqLen) + 1)) + 1
    # print startA,startB
    netInputA = np.zeros((56, 40, 5, actualSampleSeqLen), dtype=np.float32)
    netInputB = np.zeros((56, 40, 5, actualSampleSeqLen), dtype=np.float32)
    for m in range(actualSampleSeqLen):
    	img_file = os.path.join(person_sequence,"cam1",str(train_probe),image_cam1[startA+m])
    	img = cv2.imread(img_file)
    	img = cv2.resize(img,(40,56))
    	netInputA[:, :, 0, m] = img[:,:,0]
    	netInputA[:, :, 1, m] = img[:,:,1]
    	netInputA[:, :, 2, m] = img[:,:,2]
    	optical_file = os.path.join(optical_sequence,"cam1",str(train_probe),optical_cam1[startA+m])
    	optical = cv2.imread(optical_file)
    	optical = cv2.resize(optical,(40,56))
    	netInputA[:, :, 3, m] = optical[:,:,0]
    	netInputA[:, :, 4, m] = optical[:,:,1]

    for m in range(actualSampleSeqLen):
    	img_file = os.path.join(person_sequence,"cam2",str(train_gallery),image_cam2[startB+m])
    	img = cv2.imread(img_file)
    	img = cv2.resize(img,(40,56))
    	netInputB[:, :, 0, m] = img[:,:,0]
    	netInputB[:, :, 1, m] = img[:,:,1]
    	netInputB[:, :, 2, m] = img[:,:,2]
    	optical_file = os.path.join(optical_sequence,"cam2",str(train_gallery),optical_cam2[startB+m])
    	optical = cv2.imread(optical_file)
    	optical = cv2.resize(optical,(40,56))
    	netInputB[:, :, 3, m] = optical[:,:,0]
    	netInputB[:, :, 4, m] = optical[:,:,1]
    netInputA = np.transpose(netInputA, (3,2,0,1))
    netInputA = np.transpose(netInputB, (3,2,0,1))
    return netInputA,netInputB


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
    print trainID
    netInputA,netInputB = different_pair(trainID,16)
    


