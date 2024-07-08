import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class FlexibleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, num_blocks=3, base_filters=32, use_dropout=True, dropout_rate=0.5):
        super(FlexibleCNN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        channels = input_channels
        filters = base_filters
        
        for _ in range(num_blocks):
            self.layers.append(BasicBlock(channels, filters))
            channels = filters
            filters *= 2
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(channels, num_classes)
        
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        if self.use_dropout:
            x = self.dropout(x)
        
        x = self.fc(x)
        return x

