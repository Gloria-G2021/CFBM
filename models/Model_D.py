import torch.nn as nn   
import torch
import torch.nn.functional as F           

    
# model D
class CNN_D(nn.Module):
    def __init__(self):
        super(CNN_D, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding='same')
        self.conv5 = nn.Conv2d(64, 64, kernel_size=4, padding='same')
        self.conv6 = nn.Conv2d(64, 32, kernel_size=4, padding='same')
        self.conv7 = nn.Conv2d(32, 16, kernel_size=4, padding='same' )     
        self.conv8 = nn.Conv2d(16, 1, kernel_size=1, padding='same' )   
        self.conv9 = nn.Conv2d(128, 64, kernel_size=4, padding='same')
        self.conv10 = nn.Conv2d(64, 64, kernel_size=4, padding='same')
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fla = nn.Flatten()
        self.dense1 = nn.Linear(16, 64)  #(64,64)
        self.dense2 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)
        self.in1 = nn.InstanceNorm2d(64)
        self.in2 = nn.InstanceNorm2d(128)
        self.x_layer = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu(x)
        
        x = self.conv9(x)
        x = self.in1(x)
        x = self.relu(x)
        
        x = self.conv10(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv7(x)
        x = self.bn4(x)
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
    
    
