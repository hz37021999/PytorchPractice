import torch
from torchvision import transforms  #图像
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

#准备数据集   设计模型   选择loss和optimizer    训练cycle并test

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),             #   WH 28*28 -> CWH 1*28*28    C=1单通道
    transforms.Normalize((0.1307,),(0.3081,))    #均值 标准差    mnist经验值
])

#prepare dataset
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train = True,
                               download =True,
                               transform = transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size = batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train = False,
                               download =True,
                               transform = transform)
test_loader = DataLoader(test_dataset,
                          shuffle=False,
                          batch_size = batch_size)

##############################
# design model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)

    def forward(self,x):
        x = x.view(-1,784)  # N*1*28*28 -> N*784
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)   #最后一层不激活

model = Net()

criterion = torch.nn.CrossEntropyLoss() #交叉熵损失
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0): #挨个拿出训练样本
        inputs,target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  #累计loss
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f'% (epoch+1,batch_idx+1,running_loss/300))
            running_loss = 0.0

def ttest():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1) #拿最大值的下标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100*correct/total))

if __name__== '__main__':
    for epoch in range(10):
        train(epoch)
        ttest()