import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from math import floor, ceil
from model import important_variables as ivs
from PIL import Image

#Inspiration is https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


X_IMG,Y_IMG,xlist,ylist,N_ANCHORS,N_CLASSES,xanchors,yanchors,wanchors,hanchors = ivs()


#request: for image img[number].png eg img24.png, (or .jpg or whatever), there is a file img[number].txt eg img24.txt containing, for each line, five integers xi,yi,xf,yf,cn and a '\n' character where (xi,yi) are the tensor coordinates of the smallest corner of the image, and xf,yf the highest corner, and cn is the class number of whatever is in the box.

def onedim_intersection(x1,w1,x2,w2):
#return the length of the interval {x, |x-x1| <= w1/2, |x-x2| <= w2/2}
        inf1 = x1-w1/2
        sup1 = x1+w1/2
        inf2 = x2-w2/2
        sup2 = x2+w2/2
        return max(0,min(sup1,sup2)-max(inf1,inf2))

def intersection(x1,y1,w1,h1,x2,y2,w2,h2): #return the area of the intersection of the boxes defined with the two sets of coordinates
#note that the 2 dims are uncorrelated
        return onedim_intersection(x1,w1,x2,w2)*onedim_intersection(y1,h1,y2,h2)
        

def box_convert(box1):
        x1 = (box1[0]+box1[2]+1)/2
        y1 = (box1[1]+box1[3]+1)/2
        w1 = box1[2]+1-box1[0]
        h1 = box1[3]+1-box1[1]
        return x1,y1,w1,h1


def IoU(box1,box2):
        x1,y1,w1,h1 = box1
        x2,y2,w2,h2 = box2
        inter = intersection(x1,y1,w1,h1,x2,y2,w2,h2)
        union = w1*h1+w2*h2-inter
        return inter/union






def make_tensor(directory,img_index):
        target = torch.zeros(6,N_ANCHORS,xlist[-1]-2,ylist[-1]-2)
        with open(directory+"/img"+str(img_index)+".txt","r") as info:
                stringboxes = info.readlines()
                parsedboxes = [s.split(" ") for s in stringboxes]
                boxes = [(floor(X_IMG*float(a)),floor(Y_IMG*float(b)),ceil(X_IMG*float(c)),ceil(Y_IMG*float(d)),int(e[:-1])) for (a,b,c,d,e) in parsedboxes]
        ious = torch.zeros(len(boxes),N_ANCHORS,xlist[-1]-2,ylist[-1]-2)
        for na in range(N_ANCHORS):
                for i in range(xlist[-1]-2):
                        for j in range(ylist[-1]-2):
                                for b in range(len(boxes)):                                
                                        box1 = box_convert(boxes[b][:4])
                                        box2 = xanchors[na,i,j],yanchors[na,i,j],wanchors[na,i,j],hanchors[na,i,j]
                                        ious[b,na,i,j] = IoU(box1,box2)
                                        if (ious[b,na,i,j] >= 0.6):
                                                target[0,na,i,j] = (box1[0]-box2[0])/box2[2]
                                                target[1,na,i,j] = (box1[1]-box2[1])/box2[3]
                                                target[2,na,i,j] = np.log(box1[2]/box2[2])
                                                target[3,na,i,j] = np.log(box1[3]/box2[3])
                                                target[4,na,i,j] = 1
                                                target[5,na,i,j] = boxes[b][4]
                                                break
        
        for b in range(len(boxes)):
                indices_maxious = [[] for x in range(len(boxes))]
                max_ious = [0]*len(boxes)
                for na in range(N_ANCHORS):
                        for i in range(xlist[-1]-2):
                                for j in range(ylist[-1]-2):
                                        if (ious[b,na,i,j] > max_ious[b]):
                                                max_ious[b] = ious[b,na,i,j]
                                                indices_maxious[b] = [(na,i,j)]
                                        elif (ious[b,na,i,j] == max_ious[b]):
                                                indices_maxious[b].append((na,i,j))
                box1 = box_convert(boxes[b][:4])                
                for (na,i,j) in indices_maxious[b]:
                        box2 = xanchors[na,i,j],yanchors[na,i,j],wanchors[na,i,j],hanchors[na,i,j]
                        target[0,na,i,j] = (box1[0]-box2[0])/box2[2]
                        target[1,na,i,j] = (box1[1]-box2[1])/box2[3]
                        target[2,na,i,j] = np.log(box1[2]/box2[2])
                        target[3,na,i,j] = np.log(box1[3]/box2[3])
                        target[4,na,i,j] = 1
                        target[5,na,i,j] = boxes[b][4]
                                
        maxious = torch.max(ious,dim=0,keepdim=False)[0]    
        for na in range(N_ANCHORS):
                for i in range(xlist[-1]-2):
                        for j in range(ylist[-1]-2):
                                if (maxious[na,i,j] <= .3 and target[4,na,i,j] < 1):
                                        target[4,na,i,j] = -1

        torch.save(target,directory+'/tgt'+str(img_index)+'.pt')

        pos_anc = torch.nonzero(torch.where(target[4,:,:,:] == 1,torch.ones(1),torch.zeros(1)))
        Npos = pos_anc.size()[0]
        neg_anc = torch.nonzero(torch.where(target[4,:,:,:] == -1,torch.ones(1),torch.zeros(1)))
        Nneg = neg_anc.size()[0]
        if (Npos + Nneg < 256):
                print("F")
        elif (Npos < 128):
                print("Bof",Npos,Nneg)
        elif (Nneg < 128):
                print("What")
        else:
                print("Guuud")
                                

N_FILES = 48

#for i in range(N_FILES):
 #       make_tensor('dataset_test',i)

class CustomTrainingDetectionDataset(Dataset):
        def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.length = N_FILES

        def __len__(self):
                return self.length

        def __getitem__(self, idx):
                img_name = 'img'+str(idx) #or whatever naming convention
                img_fullname = self.root_dir+'/'+img_name+'.png' #if images are jpg, just change.
                image = Image.open(img_fullname).convert("RGB")
                if self.transform:
                        tensor = self.transform(image)
                
                info_img = self.root_dir+'/tgt'+str(idx)+'.pt'
                target = torch.load(info_img) 
                pos_anc = torch.nonzero(torch.where(target[4,:,:,:] == 1,torch.ones(1),torch.zeros(1)))
                Npos = pos_anc.size()[0]
                neg_anc = torch.nonzero(torch.where(target[4,:,:,:] == -1,torch.ones(1),torch.zeros(1)))
                Nneg = neg_anc.size()[0]
                #for now assume that Npos, Nneg >= 128
                target[4,:,:,:] = torch.zeros(N_ANCHORS,xlist[-1]-2,ylist[-1]-2)
                
                lpos = []
                lneg = []

                while (len(lpos) < 16 and len(lpos) < Npos):
                        x = np.random.randint(Npos)
                        if (not (x in lpos)):
                                lpos.append(x)                        

                while (len(lneg) < 32-len(lpos)):
                        x = np.random.randint(Nneg)
                        if (not (x in lneg)):
                                lneg.append(x)

                for x in lpos:
                        target[[4]+list(pos_anc[x])] = 1
                for x in lneg:
                        target[[4]+list(neg_anc[x])] = -1        
                
                return tensor,target
                
"""
idx = 0

info_img = 'dataset_test/tgt'+str(idx)+'.pt'
target = torch.load(info_img) 
pos_anc = torch.nonzero(torch.where(target[4,:,:,:] == 1,torch.ones(1),torch.zeros(1)))
Npos = pos_anc.size()[0]
neg_anc = torch.nonzero(torch.where(target[4,:,:,:] == -1,torch.ones(1),torch.zeros(1)))
Nneg = neg_anc.size()[0]
#for now assume that Npos, Nneg >= 128
target[4,:,:,:] = torch.zeros(N_ANCHORS,xlist[-1]-2,ylist[-1]-2)

lpos = []
lneg = []

print(target.size())

while (len(lpos) < 16 and len(lpos) < Npos):
        x = np.random.randint(Npos)
        if (not (x in lpos)):
                lpos.append(x)                        

while (len(lneg) < 32-len(lpos)):
        x = np.random.randint(Nneg)
        if (not (x in lneg)):
                lneg.append(x)

for x in lpos:
        print(x, end=' ')
        print(pos_anc[x],end=' ')
        target[[4]+list(pos_anc[x])] = 1
        print('t')
for x in lneg:
        print(x,end=' ')
        print(neg_anc[x])
        target[[4]+list(neg_anc[x])] = -1
        print('t') 


"""