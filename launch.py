import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable

from model import important_variables as ivs
from model import SharedVGGConvNet, SecondaryLayersRPN, FastRCNN
from data_transforms import data_transforms

from custom_dataset import CustomTrainingDetectionDataset  #custom_dataset needs to be run for image pre-treatment -- although with a bit of luck we can just split it




X_IMG,Y_IMG,xlist,ylist,N_ANCHORS,N_CLASSES,xanchors,yanchors,wanchors,hanchors = ivs()

TRAIN_BATCH_SIZE = 1

VAL_BATCH_SIZE = 256

train_loader = torch.utils.data.DataLoader(
    CustomTrainingDetectionDataset('dataset_test',
                         transform=data_transforms),
    batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=1)

#val_loader = torch.utils.data.DataLoader(
 #   datasets.ImageFolder('.' + '/val_images',
  #                       transform=data_transforms),
    #batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=1)


vggcn = SharedVGGConvNet() #ideally initialize w/ the conv layers of VGG16
optim_vgg = optim.SGD(vggcn.parameters(), lr=.001, momentum=.9)

slrpn = SecondaryLayersRPN()
optim_rpn = optim.SGD(slrpn.parameters(), lr=.01, momentum=.9)

frcnn = FastRCNN()
optim_frn = optim.SGD(frcnn.parameters(), lr=.01, momentum=.9)

Rreg = 1./1200.
lbd_rpn = 5.
Rcls = 1./256. #inverse de batch size Bcls

lbd_rcn = 5.
w_rcn = 1./200. #to balance the losses so that they are somewhat close

print("Wait")

def train_full(epoch): 
        vggcn.train()
        slrpn.train()
        frcnn.train()

        for batch_idx, (data, target) in enumerate(train_loader):
                #if use_cuda:
                 #       data, target = data.cuda(), target.cuda()
                
                optim_vgg.zero_grad()
                optim_rpn.zero_grad()
                optim_frn.zero_grad()
                print("What?")
                output1 = vggcn(data)
                #print(xlist[1:])
                out_regcls = slrpn(output1)
                out_reg = out_regcls[:,:4,:,:,:] 
                out_cls = out_regcls[:,4:,:,:,:]

                tgt_reg = target[:,:4,:,:,:] # is a grid same size as out_reg with the right stuff for positive anchors -- that is, it must be expressed with the right scale & translation invariant coordinates.
                tgt_cls_anchors = target[:,4:5,:,:,:] # is a grid same size as out_cls, but every entry is an integer 0,1,-1 (0 is "not sampled", -1 is "sampled and negative", 1 is "sampled and positive") 
                tgt_cls = target[:,5:,:,:,:] #is a grid same size as tgt_cls_anchors, but every entry is either -1 (corresponding anchor is not sampled), 0 (anchor is sampled and negative ie is background) or some class number 1 <= n <= N_CLASSES
		
                cls_crit = torch.nn.LogSigmoid()
                #print(tgt_cls_anchors.size(),out_cls.size())
                los_cls = -torch.sum(torch.where(tgt_cls_anchors == 0,torch.zeros(1),cls_crit(tgt_cls_anchors*out_cls)))

 #if not sampled, the anchor contributes 0 = log(1+0*(-1+...)), else it contributes -log(.5+target*(prediction-.5)) = -log(prediction) if it is positive and -log(1-prediction) if the anchor is negative. 

                reg_crit = torch.nn.SmoothL1Loss(reduction='none')
                #print(tgt_reg,tgt_cls_anchors,out_reg)
                los_reg = reg_crit(out_reg*torch.where(tgt_cls_anchors == 1,torch.ones(1),torch.zeros(1)),tgt_reg*torch.where(tgt_cls_anchors == 1,torch.ones(1),torch.zeros(1)))
                                                
                loss_rpn = Rcls*los_cls+lbd_rpn*Rreg*los_reg

                out_rcnn_cls, out_rcnn_box = frcnn(output1,out_regcls)
		
                out_rcnn_cls.unsqueeze(1)
                classifier = torch.nn.LogSoftmax(dim=5)
                out_rcnn_clsfieded = classifier(out_rcnn_cls)
                los_boxclas = -torch.sum(torch.where(tgt_cls_anchors != 0, out_rcnn_clsfied[:,:,:,:,:,tgt_cls],torch.zeros(1)))
		
                los_boxreg = reg_crit(out_rcnn_box*torch.where(tgt_cls >= 1,torch.ones(1),torch.zeros(1)),tgt_reg*torch.where(tgt_cls >= 1,torch.ones(1),torch.zeros(1)))

                loss = loss_rpn + w_rcn*(los_boxclas + lbd_rcn*los_boxreg)

                loss.backward()
                optim_vgg.step()
                optim_rpn.step()
                optim_frn.step()

                if batch_idx % args.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item()))

train_full(1)