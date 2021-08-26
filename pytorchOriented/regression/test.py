import torch
from torch import nn
import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


from DNN_In_Regression import test_data, MyCovid19, model_save_path

ckpt = torch.load(model_save_path+'/model.pth')
model = MyCovid19(test_data.dim)
model.load_state_dict(ckpt)

for i in range(10):
    x = torch.Tensor(test_data[i]).unsqueeze(0)
    pred = model(x)
    print(pred.item())