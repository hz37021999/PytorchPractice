import torch
import time

#use_gpu = torch.cuda.is_available()
use_gpu = False
a = time.time()

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
if(use_gpu):
    x_data = x_data.cuda()
    y_data = y_data.cuda()

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
if (use_gpu):
    model = model.cuda()
    criterion = criterion.cuda()

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
if(use_gpu):
    x_test = x_test.cuda()
y_test = model(x_test)

print('y_pred = ', y_test.data)

b = time.time()

print('total time is ',b-a)