import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable

def run(name, epochs=10, lr=1e-3, batch_size=32, use_model=False, epochsaver=1, log_interval=17):
    
    # Training settings
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='datacreation/data', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=lr, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=log_interval, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--experiment', type=str, default=name, metavar='E',
                        help='folder where experiment outputs are located.')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)
    
    # Data initialization and loading    
    from data import data_transforms
    
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data,
                             transform=data_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('datacreation/toydata',
                             transform=data_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    from model import Net
    model = Net()
    if use_model:
        namem,number = use_model
        state_dict = torch.load(namem+'/model_'+number+'.pth')
        model.load_state_dict(state_dict)
        model.eval()
        
    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')
    
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))
    
    def validation():
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            validation_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset)))
    
    for epoch in range(1, args.epochs + 1):
        if epoch%10==0:
            lr*=0.8
            optimizer = torch.optim.Adam(model.parameters(), lr)
        train(epoch)
        validation()
        if epoch%epochsaver==0:
            model_file = args.experiment + '/model_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), model_file)
            #print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')

