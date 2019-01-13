import zipfile
import os

import torchvision.transforms as transforms
from model import important_variables as ivs

X_IMG,Y_IMG,xlist,ylist,N_ANCHORS,N_CLASSES,xanchors,yanchors,wanchors,hanchors = ivs()


# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.Resize((X_IMG, Y_IMG)),
    transforms.ToTensor()])
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 #std=[0.229, 0.224, 0.225])


