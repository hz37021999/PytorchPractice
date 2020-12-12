import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

'''
in_channels, out_channels = 5,10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size,in_channels,width,height)
conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)
'''

'''
torch.Size([1, 5, 100, 100]) 输入5个通道100*100
torch.Size([1, 10, 98, 98])  输出10个通道100-3+1  98*98
torch.Size([10, 5, 3, 3])    输出10 输入5 卷积和3*3   必要的四个值
'''
'''
1*28*28-- conv2d k=5 -->10*24*24-- relu -->10*24*24-- pooling k=2 -->10*12*12 --
conv2d k=5 --> 20*8*8 --> relu --> 20*8*8 --> pooling k=2 -->20*4*4-->view-->320--linear-->10
'''

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),   #convert PIL image to Tensor  Channel*W*H   28*28 --> 1*28*28
    transforms.Normalize((0.1307,),(0.3081,))  #切换到01分布 mean  std   mnist所有像素的均值和标准差
])

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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=3,padding=True)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=3,padding=True)
        self.conv3 = torch.nn.Conv2d(20,40,kernel_size=3,padding=True)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(360, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 10)


    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = self.pooling(F.relu(self.conv3(x)))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1,batch_idx+1,running_loss/2000))
            running_loss = 0.0

def ttest():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,dim=1)  #最大值下标取出来 沿着第一个维度找 行/列？
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy on test set: %d %% [%d %d]' % (100*correct/total, correct, total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        ttest()
