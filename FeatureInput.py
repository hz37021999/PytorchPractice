import torchvision
import torch
import torch.nn.functional as F   #sigmoid relu 等等
import numpy as np

x = np.loadtxt('X.csv', delimiter=' ',dtype=np.float32)
y = np.loadtxt('y.csv', delimiter=None,dtype=np.float32)
print(np.shape(x))
print(np.shape(y))
x_data = torch.from_numpy(x[:,:])
print(x_data.shape)
y_data = torch.from_numpy(y[:,])
y_data = y_data.reshape(442,1)
print(y_data.shape)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(10,4)
        self.linear2 = torch.nn.Linear(4,2)
        self.linear3 = torch.nn.Linear(2,1)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x

model = Model()
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



