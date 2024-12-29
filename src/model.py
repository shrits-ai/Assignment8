import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        # Initial Conv Layer
        self.initial_conv = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)  # RF: 3x3
        self.bn1 = nn.BatchNorm2d(12)
        
        # Depthwise Separable and Dilated Conv 
        # Block 1 
        self.depthwise1 = nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1, groups=12)  # RF: 5x5
        self.pointwise1 = nn.Conv2d(12, 16, kernel_size=1, stride=1)  # RF: 5x5 (no change)
        self.bn2 = nn.BatchNorm2d(16)
        self.dilated_conv1 = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=2, dilation=2)  # RF: 9x9
        self.bn3 = nn.BatchNorm2d(24)
        self.downsample = nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1)  # RF: 13x13
        self.bn4 = nn.BatchNorm2d(32)
        
        # Depthwise Separable and Dilated Conv 
        # Block 2
        self.depthwise2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, groups=32, dilation=2)  # RF: 17x17
        self.pointwise2 = nn.Conv2d(32, 48, kernel_size=1, stride=1)  # RF: 17x17 (no change)
        self.bn5 = nn.BatchNorm2d(48)
        self.dilated_conv2 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=4, dilation=4)  # RF: 25x25
        self.bn6 = nn.BatchNorm2d(64)

        self.depthwise3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=8, groups=64, dilation=8)  # RF: 41x41
        self.pointwise3 = nn.Conv2d(64, 128, kernel_size=1, stride=1)  # RF: 41x41 (no change)
        self.bn7 = nn.BatchNorm2d(128)
        self.dilated_conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)  # RF: 45x45
        self.bn8 = nn.BatchNorm2d(128)
        
        # Final Layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # RF: Covers the entire input
        self.dropout = nn.Dropout(0.05)
        self.fc = nn.Linear(128, NUM_CLASSES)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.initial_conv(x)))  # RF: 3x3
        x = F.relu(self.bn2(self.pointwise1(self.depthwise1(x))))  # RF: 5x5
        x = F.relu(self.bn3(self.dilated_conv1(x)))  # RF: 9x9
        x = F.relu(self.bn4(self.downsample(x)))  # RF: 13x13
        x = F.relu(self.bn5(self.pointwise2(self.depthwise2(x))))  # RF: 17x17
        x = F.relu(self.bn6(self.dilated_conv2(x)))  # RF: 25x25
        x = F.relu(self.bn7(self.pointwise3(self.depthwise3(x))))  # RF: 41x41
        x = F.relu(self.bn8(self.dilated_conv3(x)))  # RF: 45x45
        x = self.global_avg_pool(x)  # RF: Covers the entire input
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
