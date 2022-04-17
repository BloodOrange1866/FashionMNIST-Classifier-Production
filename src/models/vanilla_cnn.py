import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=(1,1))
        self.bn1 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1))
        self.bn2 = nn.BatchNorm2d(32)
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = x.unsqueeze(1) # for input BxCinxHxW
        x = x.to(torch.float)

        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp(x)

        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.mp(x)

        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x