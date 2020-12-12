import torch
from torch import nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
'''
The Inception Module of GoogleNet
concatenate (1)--> avepooling --> 1*1 conv            --> concatenate 
            (2)--> 1*1 conv                           -->
            (3)--> 1*1 conv --> 5*5 conv              -->
            (4)--> 1*1 conv --> 3*3 conv --> 3*3 conv -->
            
(1) self.branch_pool = nn.Conv2d(in_channels,24,kernel_size=1)
    branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) #大小不变
    branch_pool = self.branch_pool(branch_pool)
(2) self.branch1x1 = nn.Conv2d(in_channels,16,kernel_size=1)
    branch1x1 = self.branch1x1(x)
(3) self.branch5x5_1 = nn.Conv2d(in_channels,16,kernel_size=1)
    self.branch5x5_2 = nn.Conv2d(16,24,kernel_size=5,padding=2)
    branch5x5 = self.branch5x5_1(x)
    branch5x5 = self.branch5x5_2(branch5x5)
(4) self.branch3x3_1 = nn.Conv2d(in_channels,16,kernel_size=1)
    self.branch3x3_2 = nn.Conv2d(16,24,kernel_size=3,padding=1)
    self.branch3x3_3 = nn.Conv2d(24,24,kernel_size=3,padding=1)
    branch3x3 = self.branch3x3_1(x)
    branch3x3 = self.branch3x3_2(branch3x3)
    branch3x3 = self.branch3x3_3(branch3x3)
    
outputs = [branch1x1,branch5x5,branch3x3,branch_pool]
return torch.cat(outputs,dim=1)
'''

class InceptionA(nn.Module):
    def __init__(self,in_channels):
        super(InceptionA, self).__init__()
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self,x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)  # 大小不变
        branch_pool = self.branch_pool(branch_pool)

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(88,20,kernel_size=5)

        self.incept1 = InceptionA(in_channels=10)
        self.incept2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408,10)  #

    def forward(self,x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incept1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incept2(x)
        x = x.view(in_size,-1) #
        x = self.fc(x)         # 可删掉看输出
        return x




