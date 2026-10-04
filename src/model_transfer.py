import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=2):
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet
