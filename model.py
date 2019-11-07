import torch
import torch.nn as nn
from torchstat import stat
from torchsummary import summary
from math import sqrt

class UDA(nn.Module):
    def __init__(self,channels):#图片在这里时，也是B*C*H*W的结构
        super(UDA,self).__init__()
        self.channels=channels
        self.conv1=nn.Conv2d(in_channels=3,out_channels=channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv3=nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.fc=nn.Linear(16*channels,10,bias=False)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,sqrt(2./n))
    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.relu(self.conv3(x))
        x=x.view((-1,16*self.channels))
        return self.fc(x)
        
if __name__=='__main__':
    net=UDA(64)       
    stat(net, (3, 32, 32))

    #summary(net, (3, 32, 32)) 不知道为什么 它不可以测这个网络的信息