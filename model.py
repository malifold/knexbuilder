import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

#TO DO: 

#Implement RoI pooling -- hopefully done.
#Implement non-max suppression wherever the authors intended it to be (hopefully without it messing too much with potential backpropagation) 
#Implement Fast RCNN itself training -- hopefully done
#I found something at URL https://github.com/ruotianluo/pytorch-faster-rcnn


#######################################
#                                     #
#  SHARED CONVOLUTIONAL NETWORK MODEL #
#                                     #
#######################################



###
# Neural model itself : VGG 16
###


class SharedVGGConvNet(nn.Module):
        def __init__(self):
                super(SharedVGGConvNet, self).__init__()
                self.conv1a = nn.Conv2d(3, 64, kernel_size=3) 
                self.conv1b = nn.Conv2d(64, 64, kernel_size=3) 
                self.conv2a = nn.Conv2d(64, 128, kernel_size=3)
                self.conv2b = nn.Conv2d(128, 128, kernel_size=3)
                self.conv3a = nn.Conv2d(128, 256, kernel_size=3)
                self.conv3b = nn.Conv2d(256, 256, kernel_size=3)
                self.conv3c = nn.Conv2d(256, 256, kernel_size=3)
                self.conv4a = nn.Conv2d(256, 512, kernel_size=3) 
                self.conv4b = nn.Conv2d(512, 512, kernel_size=3) 
                self.conv4c = nn.Conv2d(512, 512, kernel_size=3)
                self.conv5a = nn.Conv2d(512, 512, kernel_size=3) 
                self.conv5b = nn.Conv2d(512, 512, kernel_size=3) 
                self.conv5c = nn.Conv2d(512, 512, kernel_size=3)

        def forward(self, x):
                x = F.relu(self.conv1a(x))
                x = F.relu(self.conv1b(x))
                x = F.max_pool2d(x,2)
                x = F.relu(self.conv2a(x))
                x = F.relu(self.conv2b(x))
                x = F.max_pool2d(x,2)
                x = F.relu(self.conv3a(x))
                x = F.relu(self.conv3b(x))
                x = F.relu(self.conv3c(x))
                x = F.max_pool2d(x,2)
                x = F.relu(self.conv4a(x))
                x = F.relu(self.conv4b(x))
                x = F.relu(self.conv4c(x))
                x = F.max_pool2d(x,2)
                x = F.relu(self.conv5a(x))
                x = F.relu(self.conv5b(x))
                x = F.relu(self.conv5c(x))
                return x

###
# Grid transformation through the layers
###
oplist = ['conv','conv','MP','conv','conv','MP','conv','conv','conv','MP','conv','conv','conv','MP','conv','conv','conv']


"""paramlist = [(if (x == 'conv') then 3 else 2) for x in oplist]

if ever we decided to change the conv net architecture, and use non 2x2 maxpools or non 3x3 conv kernels ... I did not leave strides possible though but let's worry about that later"""



X_IMG = 720
Y_IMG = 720

xlist = [X_IMG]
ylist = [Y_IMG]
for op in oplist:
        if (op == 'conv'):
                xlist.append(xlist[-1]-2)
                ylist.append(ylist[-1]-2)
        else:
                xlist.append((xlist[-1])//2)
                ylist.append((ylist[-1])//2)


"""
This was a test code.
mns = torch.zeros((1,3,600,1000))
x = torch.normal(mns,1.0)

n = SharedVGGConvNet()
import datetime
time1 = datetime.datetime.now()
y = n(x)
time2 = datetime.datetime.now()
print(y.size())
print(y[0:,:,:,:].size())
print(time1)
print(time2)"""











###############
#             #
#  RPN MODULE #
#             #
###############




###
# RPN Module
###

N_ANCHORS = 25

class SecondaryLayersRPN(nn.Module):
        def __init__(self):
                super(SecondaryLayersRPN, self).__init__()
                self.conv = nn.Conv2d(512,512,kernel_size=3)
                self.regcls = nn.Conv2d(512,5*N_ANCHORS,kernel_size=1) #as suggested in the footnote cls uses logistic regression and outputs one score per anchor
                #self.cls = nn.Conv2d(512,2*N_ANCHORS,kernel_size=1)
        def forward(self,x):
                y = F.relu(self.conv(x))
                y = self.regcls(y)
                z1 = y[:,:N_ANCHORS,:,:]
                z2 = y[:,N_ANCHORS:2*N_ANCHORS,:,:]
                z3 = y[:,2*N_ANCHORS:3*N_ANCHORS,:,:]
                z4 = y[:,3*N_ANCHORS:4*N_ANCHORS,:,:]
                z5 = y[:,4*N_ANCHORS:,:,:]
                z = torch.stack((z1,z2,z3,z4,z5), 1)
                return z 





###
# Anchor design
### 


#THIS COMMENT IS IRRELEVANT NOW
#anchors are chosen with three possible base sizes and three possible aspect rationes (latin plural of ratio?) at each location in the grid after the common conv layer of the last network: 
#base sizes: 96 x 96, 192 x 192, 384 x 384
#aspect ratios: 1/1, 2/1, 1/2, .5/3, 3/.5 

#x and y are coordinates of the centers of the anchors -- they are in the actual image, if I understood correctly -- so I guess I'll just take them to be the centers of the "influence zone" of the window and pray for a correct computation.


def window_coords(x0,y0): #x0 and y0 are coordinates of a "pixel" on the convolutional feature map after the first convolutional layer of the SecondaryLayerRPN
        xi,yi,xf,yf = x0,y0,x0+1,y0+1
        xi,yi,xf,yf = xi,yi,xf+2,yf+2 #going back through the first conv layer of SL_RPN
        for i in range(len(oplist)-1,0,-1):
                if oplist[i] == 'conv':
                        xi,yi = xi,yi
                        xf,yf = xf+2,yf+2
                else:
                        xi,yi = 2*xi,2*yi
                        xf,yf = 2*xf,2*yf
                        """if (xf > xlist[i]):
                                xf -= 1
                        if (yf > ylist[i]):
                                yf -= 1""" #I misunderstood the way maxpool2d works
        return xi,yi,xf,yf

def window_center(x0,y0):
        xi,yi,xf,yf = window_coords(x0,y0)
        return (float(xi)+float(xf))/2.,(float(yi)+float(yf))/2.

xanchors = torch.zeros(N_ANCHORS,xlist[-1]-2,ylist[-1]-2)
yanchors = torch.zeros(N_ANCHORS,xlist[-1]-2,ylist[-1]-2)
wanchors = torch.zeros(N_ANCHORS,xlist[-1]-2,ylist[-1]-2)
hanchors = torch.zeros(N_ANCHORS,xlist[-1]-2,ylist[-1]-2)

 #same grid size as z in SL RPN above -- 


"""
64x64
64x128
128x64
128x128
128x256
256x128
256x256
256x512
512x256
512x512
64x96
96x64
64x192
192x64
128x192
192x128
256x384
384x256
64x320
320x64
48x480
480x48
480x96
96x480
"""


list_xanchors = [64,64,128,128,128,256,256,256,512,512,64,96,64,192,128,192,256,384,64,320,48,480,480,96]
list_yanchors = [64,128,64,128,256,128,256,512,256,512,96,64,192,64,192,128,384,256,320,64,480,48,96,480]

for i in range(len(list_xanchors)):
        wanchors[i,:,:] = list_xanchors[i]
        hanchors[i,:,:] = list_yanchors[i]
"""for i in range(3):
        basesize = 96
        height,width = ((i+1)//2),((i+1)%2)
        if (height*width == 0):
                height += 1
                width += 1
        for j in range(3):
                wanchors[3*j+i,:,:] = width*basesize
                hanchors[3*j+i,:,:] = height*basesize 
                basesize *= 2
wanchors[9,:,:] = 3*96
hanchors[9,:,:] = 48
hanchors[10,:,:] = 3*96
wanchors[10,:,:] = 48
wanchors[12,:,:] = 3*192
hanchors[12,:,:] = 96
hanchors[11,:,:] = 3*192
wanchors[11,:,:] = 96
"""
for i in range(xlist[-1]-2):
        for j in range(ylist[-1]-2):
                xanchors[:,i,j],yanchors[:,i,j] = window_center(i,j)
                #print(xanchors[0,i,j],yanchors[0,i,j])        



##############################
#                            #
# FAST RCNN DETECTION MODULE #
#                            #
##############################


N_CLASSES = 16 #value unsure -- does not include background
ROI_THRESHOLD = 0.001 #some regions just are pointless as RoI

class FastRCNN(nn.Module):
        def __init__(self):
                super(FastRCNN,self).__init__()
                self.Height = 7
                self.Width = 7
                self.fc1 = nn.Linear(512*49,200)		#200 is random, 49=7*7 is suggested in the article Fast RCNN for reasons I do not understand
                self.fc2 = nn.Linear(200,100)		#100 is also random			
                self.fc3 = nn.Linear(100,N_CLASSES+1) #for type classifier + box regressor
                self.fc4 = nn.Linear(100,4*(N_CLASSES+1)) #box regressor
	
        def forward(self,cfeats,regcls): #cfeats = convolutional features = entry of 2ndary layer RPN ; regcls = output of 2ndary layer
                regcls.detach() #we do not want any gradient backpropagation through regcls, ie regcls is supposed to be a given constant at this point
                xreg = regcls[:,0,:,:,:]
                yreg = regcls[:,1,:,:,:]
                wreg = torch.exp(regcls[:,2,:,:,:])*wanchors
                hreg = torch.exp(regcls[:,3,:,:,:])*hanchors
                xreg = xreg*wanchors+xanchors  
                yreg = yreg*hanchors+yanchors
                xis = xreg-(wreg/2.)
                yis = yreg-(hreg/2.)
                xfs = xreg+(wreg/2.)
                yfs = yreg+(hreg/2.) #computation of the NW/SE corners of the grids

                conf = torch.zeros(regcls.size())
                conf.detach()
                conf[:,0,:,:,:] = torch.floor(F.relu(xis))
                conf[:,1,:,:,:] = torch.floor(F.relu(yis))
                conf[:,2,:,:,:] = torch.ceil(xfs)
                conf[:,3,:,:,:] = torch.ceil(yfs)
                conf[:,2,:,:,:] = torch.min(conf[:,2,:,:,:],(X_IMG-1)*torch.ones(1))
                conf[:,3,:,:,:] = torch.max(conf[:,3,:,:,:],(Y_IMG-1)*torch.ones(1))
                conf[:,4,:,:,:] = torch.where((xis < X_IMG*torch.ones(1)) & (yis < Y_IMG*torch.ones(1)) & (xfs >= torch.zeros(1)) & (yfs >= torch.zeros(1)),torch.nn.Sigmoid()(regcls[:,4,:,:,:]),torch.zeros(1)) #they are adapted to the borders of the image -- windows that are completely out of the picture are given a confidence score of 0, whatever the prediction by the RPN

#I have a terrifying feeling that at first try all confidence scores will be zero for this reason.

# Propagation of the RPN proposed boxes through the VGG conv layer -- actual meaning is unclear to me, so I computed the area of the convolutional map that could have been affected; there us no further RoI selection, except that regions outside the image are penalized with a zero confidence score -- see after
                
                for i in range(len(oplist)): 
                        if (oplist[i] == 'conv'):
                                conf[:,:2,:,:,:] = F.relu(conf[:,:2,:,:,:]-1)
                                conf[:,2,:,:,:] = torch.min(conf[:,2,:,:,:],(xlist[i+1]-1)*torch.ones(1))
                                conf[:,3,:,:,:] = torch.min(conf[:,3,:,:,:],(ylist[i+1]-1)*torch.ones(1))
                        else:
                                conf[:,:4,:,:,:] = torch.floor(conf[:,:4,:,:,:]/2.)
                #finalrois = torch.nonzero(F.relu(conf[:,4,:,:,:]-ROI_THRESHOLD)) 

                #roipooled = torch.zeros(finalrois.size()[0],512,49)
 
                #RoI Pooling -- if we could tensorize that stuff it would be great because this might improve speed.
                roipooled = torch.zeros(conf.size()[0],N_ANCHORS,xlist[-1]-2,ylist[-1]-2,512,49)
                for roi_index in range(conf.size()[0]*N_ANCHORS*(xlist[-1]-2)*(ylist[-1]-2)):  
                        img_index,rmd1 = roi_index // (N_ANCHORS*(xlist[-1]-2)*(ylist[-1]-2)),roi_index % (N_ANCHORS*(xlist[-1]-2)*(ylist[-1]-2))
                        conf_anchor,rmd2 = rmd1 // ((xlist[-1]-2)*(ylist[-1]-2)), rmd1 % ((xlist[-1]-2)*(ylist[-1]-2))                         
                        conf_x,conf_y = rmd2 // (ylist[-1]-2), rmd2 % (ylist[-1]-2)
                        
                        roi_xi = int(conf[img_index,0,conf_anchor,conf_x,conf_y])
                        roi_yi = int(conf[img_index,1,conf_anchor,conf_x,conf_y])
                        roi_xf = int(conf[img_index,2,conf_anchor,conf_x,conf_y])
                        roi_yf = int(conf[img_index,3,conf_anchor,conf_x,conf_y])
                        print("\t\t"+str(roi_xi)+" "+str(roi_yi)+" "+str(roi_xf)+" "+str(roi_yf))
                        if (conf[img_index,4,conf_anchor,conf_x,conf_y] > 0):
                                for i in range(7):
                                        xi_piece = int(roi_xi + (i*(roi_xf+1-roi_xi))//self.Width)
                                        xf_piece = int(roi_xi + ((i+1)*(roi_xf+1-roi_xi))//self.Width)
                                        #print(img_index,xi_piece,xf_piece)
                                        if (xi_piece < xf_piece):
                                                interm = torch.max(cfeats[img_index,:,xi_piece:xf_piece,:],dim=1,keepdim=False)[0]
                                        #print(cfeats[img_index,:,xi_piece:xf_piece,:].size(),interm.size())
                                                for j in range(7):
                                                        yi_piece = int(roi_yi + (j*(roi_yf+1-roi_yi))//self.Height)
                                                        yf_piece = int(roi_yi + ((j+1)*(roi_yf+1-roi_yi))//self.Height)
                                                        #print(yi_piece,yf_piece,roi_yi,roi_yf,interm[:,yi_piece:yf_piece].size())                                                
                                                        if (yi_piece < yf_piece):                                                
                                                                roipooled[img_index,conf_anchor,conf_x,conf_y,:,7*i+j] = torch.max(interm[:,yi_piece:yf_piece],dim=1,keepdim=False)[0]
                                                        else:
                                                                roipooled[img_index,conf_anchor,conf_x,conf_y,:,7*i+j] = 0
                                        else:
                                                for j in range(7):
                                                        roipooled[img_index,conf_anchor,conf_x,conf_y,:,7*i+j] = 0
                        roipooled.view(conf.size()[0],N_ANCHORS,xlist[-1]-2,ylist[-1]-2,-1)

                z = F.relu(self.fc1(roipooled))
                z = F.relu(self.fc2(z))
                z1 = self.fc3(z)
                z2 = self.fc4(z) #now we rearrange z2 so that its shape "matches" the wished shape for tgt_reg, see after
                z2_0 = z2[:,:,:,:,:N_CLASSES+1]
                z2_1 = z2[:,:,:,:,N_CLASSES+1:2*(N_CLASSES+1)]
                z2_2 = z2[:,:,:,:,2*(N_CLASSES+1):3*(N_CLASSES+1)]
                z2_3 = z2[:,:,:,:,3*(N_CLASSES+1):]
                z3 = torch.stack((z2_0,z2_1,z2_2,z2_3),1)
                return z1,z3




def important_variables():
        return X_IMG,Y_IMG,xlist,ylist,N_ANCHORS,N_CLASSES,xanchors,yanchors,wanchors,hanchors
