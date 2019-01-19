import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 16

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=2, stride=2) #10*32*32
        self.conv2 = nn.Conv2d(10, 20, kernel_size=2, stride=2) #20*16*16
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, padding=1) #20*16*16->8*8
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3, padding=1) #20*8*8->4*4
        self.conv5 = nn.Conv2d(20, 20, kernel_size=2, stride=2) #20*2*2
        self.fc1 = nn.Linear(20*2*2, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(self.conv2(x), 2)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.relu(self.conv5(x), 2)
        x = x.view(-1, 20*2*2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)




class Inception(nn.Module):
    def __init__(self,cin,cout):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=1) #1
        self.conv2 = nn.Conv2d(cin, cout, kernel_size=1) #1-
        self.conv3 = nn.Conv2d(cout, cout, kernel_size=3, padding=1) #-3
        self.conv4 = nn.Conv2d(cin, cout, kernel_size=1) #1-
        self.conv5 = nn.Conv2d(cout, cout, kernel_size=5, padding=2) #-5
        self.conv6 = nn.Conv2d(cin, cout, kernel_size=1) #-1
        self.bn = nn.BatchNorm2d(cout*4)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv3(F.relu(self.conv2(x))))
        x3 = F.relu(self.conv5(F.relu(self.conv4(x))))
        x4 = F.relu(self.conv6(F.max_pool2d(x,3,stride=1,padding=1)))
        x = torch.cat([x1,x2,x3,x4],1)
        return self.bn(F.relu(x))

class IncNet(nn.Module):
    def __init__(self):
        super(IncNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, padding=1, stride=2) #60
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding=1) #30
        self.inc1 = Inception(32,8)
        self.inc2 = Inception(32,8)
        self.conv3 = nn.Conv2d(32,64,kernel_size=3, padding=1) #30
        self.inc3 = Inception(64,16)
        self.inc4 = Inception(64,16)
        self.conv4 = nn.Conv2d(64,128,kernel_size=3, padding=1) #15
        self.fc1 = nn.Linear(128*5*5, 100)
        self.fc2 = nn.Linear(100, nclasses)

    def forward(self, x):
        x = F.local_response_norm(F.relu(F.max_pool2d(self.conv1(x),2)),2)
        x = F.relu(self.conv2(x))
        x = self.inc2(self.inc1(x))
        x = F.local_response_norm(F.relu(F.max_pool2d(self.conv3(x),2)),2)
        x = self.inc4(self.inc3(x))
        x = F.relu(F.avg_pool2d(self.conv4(x),3))
        x = F.dropout(x.view(-1, 128*5*5),p=0.5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        
        
        

class Resblock(nn.Module):
    def __init__(self,c):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(c,c, kernel_size=3, padding=1) #1
        self.conv2 = nn.Conv2d(c,c, kernel_size=3, padding=1) #1
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        y = self.conv2(F.relu(self.conv1(x)))
        return self.bn(F.relu(x+y))

class Resdown(nn.Module):
    def __init__(self,c):
        super(Resdown, self).__init__()
        self.conv1 = nn.Conv2d(c,c*2, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(c*2,c*2, kernel_size=3, padding=1)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, padding=3, stride=2) #64
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding=1) #32
        self.rb1 = Resblock(32)
        self.rb2 = Resblock(32)
        self.rd1 = Resdown(32) #16
        self.rb3 = Resblock(64)
        self.rb4 = Resblock(64)
        self.rd2 = Resdown(64) #8
        self.rb5 = Resblock(128)
        self.rb6 = Resblock(128)
        self.fc1 = nn.Linear(128*4*4,200)
        self.fc2 = nn.Linear(200,20)

    def forward(self, x):
        x = F.local_response_norm(F.relu(F.max_pool2d(self.conv1(x),2)),2)
        x = F.relu(self.conv2(x))
        x = self.rd1(self.rb2(self.rb1(x)))
        x = self.rd2(self.rb4(self.rb3(x)))
        x = F.relu(F.avg_pool2d(self.rb6(self.rb5(x)),2))
        x = F.dropout(x.view(-1, 128*4*4),p=0.3)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


"""
4. BLACK
9. BLACK
10. BLACK with RED SPOT
11. BLACK/BROWN
12. BLACK with YELLOW HEAD
13. BLACK with pale yellow/white feathers
14. INDIGO
15. BLUE with pale yellow/white feathers
16. RGB spots
19. Gray
20. Gray/yellow chest
21. Black with rusty streak
23. Black
26. Black
28. bark/white
29. Black
30. Black
31. Brown/white
33. Brown/white
34. rust w/ gray head

4,9,11,21,23,26,29,30
10
11,21
12
13
14
15
16
19
20
28,31,33
34
"""


class Color(nn.Module):
    def __init__(self):
        super(Color, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3) #10*62*62
        self.conv2 = nn.Conv2d(6, 6, kernel_size=5) #24*27*27
        self.conv3 = nn.Conv2d(6, 6, kernel_size=3) #24*7*7
        self.fc1 = nn.Linear(6*7*7, 32)
        self.fc2 = nn.Linear(32, ncolors)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) #10*31*31
        x = F.relu(F.max_pool2d(self.conv2(x), 3)) #24*9*9
        x = F.relu(self.conv3(x)) #24*7*7
        x = x.view(-1, 6*7*7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)