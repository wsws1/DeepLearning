import torch
from torch import nn
import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wk_dir = '../data/lhyHW/HW1/'
epoch_num = 20
lr = 0.001
model_save_path = wk_dir + 'model'
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
class MyDataset(Dataset):

    def __init__(self, mode='train', wk_dir=''):
        super(MyDataset, self).__init__()
        self.mode = mode
        train_data_path = wk_dir + 'covid.train.csv'
        test_data_path = wk_dir + 'covid.test.csv'
        if mode == 'train':
            path = train_data_path
            with open(path, 'r') as fp:
                data = list(csv.reader(fp))
                data = np.array(data[1:])[:, 1:].astype(float)
            self.target = data[:, -1]
            self.data = data[:, : -1]
        else:
            path = test_data_path
            with open(path, 'r') as fp:
                data = list(csv.reader(fp))
                data = np.array(data[1:])[:, 1:].astype(float)
            self.data = data

        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(axis=0, keepdims=True)) \
            / self.data[:, 40:].std(axis=0, keepdims=True)

        self.dim = self.data.shape[1]

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

train_data = MyDataset(wk_dir=wk_dir)
test_data = MyDataset(wk_dir=wk_dir, mode='test')
data_loader = DataLoader(train_data, batch_size=100, shuffle=True)
class MyCovid19(nn.Module):

    def __init__(self, input_dim):
        super(MyCovid19, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


model = MyCovid19(train_data.dim).double()
criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)


if __name__ == '__main__':
    for epoch in range(epoch_num):
        optimizer.zero_grad()
        total_loss = 0
        for i, data in enumerate(data_loader):
            x, y = data
            bs = x.shape[0]
            # print(x, y)
            out = model(x)
            loss = criterion(out, y)
            cur_loss = loss.item()
            total_loss += cur_loss
            loss.backward()
            optimizer.step()
        print("epoch ", epoch, "loss ", total_loss / len(data_loader) / bs * 100.0)

    print("save model")
    torch.save(model.state_dict(), model_save_path + '/model.pth')



