#DataSet & DataLoader
#epoch 所有样本全都进行了一次forward和backward
#batch-size 一次forward和backward用的样本数量
#iteration batch-size的个数

import torch
import numpy as np
from torch.utils.data import Dataset #抽象类
from torch.utils.data import DataLoader

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, item): #之后支持下标操作
        return self.x_data[item],self.y_data[item]

    def __len__(self):  #条数
        return self.len

dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8,4)
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

if __name__ == '__main__':
    for epoch in range(100):
        for i,data in enumerate(train_loader,0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            print(epoch,i,loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



