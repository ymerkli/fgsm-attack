import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN architecture for MNIST classification
class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1) # output is 26x26x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1) # output is 24x24x64
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(in_features=9216, out_features=128) # in_features = 24*24*64*0.25
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output