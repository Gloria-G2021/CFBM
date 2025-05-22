import torch.nn as nn   
import torch
import torch.nn.functional as F           

# Model A
class CNN_A(nn.Module):
    def __init__(self):
        super(CNN_A, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, padding='same')    
        self.conv8 = nn.Conv2d(16, 1, kernel_size=1, padding='same' )   
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fla = nn.Flatten()
        self.dense1 = nn.Linear(16, 64)  #(64,64)
        self.dense2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.in1 = nn.InstanceNorm2d(64)
        self.x_layer = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        
        x = self.conv8(x)
        x = self.relu(x)
    
        x = self.pool1(x)
        
        x = self.fla(x)
        
        x = self.dense1(x)
        x = self.relu(x)
        
        x = self.dense2(x)
        x = self.relu(x)
        
        x = self.fla(x)
        x = self.x_layer(x)
        
        x = self.softmax(x)
        
        return x
        
        
