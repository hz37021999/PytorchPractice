import torchvision
import torch
import torch.nn.functional as F   #sigmoid relu 等等
import numpy as np
import matplotlib.pyplot as plt
train_set = torchvision.datasets.MNIST(root='../dataset/mnist',train=True, download=False)
test_set = torchvision.datasets.MNIST(root='../dataset/mnist',train=False, download=False)

#准备数据
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

#模型构造
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.liner = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.liner(x))
        return y_pred
model = LogisticRegressionModel()

#定义loss和optimizer
#加了回归之后，比较分布的差异 cross-entropy 交叉熵
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

#训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0,10,200)
x_t = torch.Tensor(x).view((200,1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()