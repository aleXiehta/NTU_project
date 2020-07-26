import torch
import torch.nn as nn
from pretrainedmodels import xception

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception = xception()
        self.xception.eval()
        for p in self.xception.parameters():
            p.requires_grad = False
        self.cnn1 = nn.Conv2d(
            in_channels=2048,
            out_channels=512,
            kernel_size=3,
            stride=1
        )
        self.bn = nn.BatchNorm2d(num_features=512)

        # B x 256 x 4 x 4
        self.fc1 = nn.Linear(18432, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.xception.features(x)
        x = self.cnn1(x)
        x = self.bn(x)
        x = self.fc1(x.flatten(start_dim=1))
        x = self.fc2(x.relu())
        return x.squeeze(-1)
