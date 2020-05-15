# architechtures
import torch.nn as nn
import torch.nn.functional as F
def CNN_model(n_channels=1):
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            
            self.downsample = nn.AvgPool3d(2, stride=2, padding=0)
            
            self.CNNlayer = nn.Sequential(
                nn.Conv3d(n_channels, 64, kernel_size=3, stride=1),
                nn.ELU(),
                nn.Conv3d(64, 16, kernel_size=3, stride=1),
                nn.ELU(),
                nn.MaxPool3d(2)
                )
            
            self.flat1 = nn.Linear(14256, 16)        
            
            self.flat2 = nn.Linear(16, 1)
                
        def forward(self, x):
            
            x=self.downsample(x)
            x=self.CNNlayer(x)
            x= x.reshape(x.size(0), -1)   
            
            x=F.elu(self.flat1(x))
            x=self.flat2(x)
            
            return x    