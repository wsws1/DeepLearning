import torch
from torchvision import models
from torch import nn

class resnet18(nn.Module):

    def __init__(self):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_feature = self.fc.in_features