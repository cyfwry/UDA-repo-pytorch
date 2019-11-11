import torch
import torch.nn as nn
from math import sqrt
class res_block(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(res_block,self).__init__()
        self.batchnorm_share=nn.BatchNorm2d(out_channels) #share?CRB or CBR?
        self.left=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1,bias=False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,stride=stride,bias=False),
                                nn.BatchNorm2d(out_channels))
        self.relu=nn.ReLU(inplace=True)
        self.shortcut=nn.Sequential() #这样子就是什么都不做
        if stride!=1 or in_channels!=out_channels:
            self.shortcut=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,bias=False),
                                        nn.BatchNorm2d(out_channels))
            
    def forward(self,x):
        out=self.left(x)
        out+=self.shortcut(x)
        out=self.relu(out)
        return out
        
class resnet18(nn.Module):
    def __init__(self):#图片在这里时，也是B*C*H*W的结构
        super(resnet18,self).__init__()
        self.input=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1,bias=False),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True))
        self.avepool=nn.AvgPool2d(kernel_size=2,stride=2)
        self.res_block1=self.make_layer(res_block,in_channels=64,out_channels=128,stride=2,nums=2)
        self.res_block2=self.make_layer(res_block,in_channels=128,out_channels=256,stride=2,nums=2)
        self.res_block3=self.make_layer(res_block,in_channels=256,out_channels=512,stride=2,nums=2)
        self.res_block4=self.make_layer(res_block,in_channels=512,out_channels=512,stride=1,nums=2)
        self.fc=nn.Linear(2048,10)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,sqrt(2./n))
                
    def make_layer(self,block,in_channels,out_channels,stride,nums):
        layers=[]
        for i in range(nums-1):
            layers.append(block(in_channels,in_channels,1))
        layers.append(block(in_channels,out_channels,stride))
        return nn.Sequential(*layers)
        
    def forward(self,x):
        x=self.input(x)
        x=self.res_block1(x)
        x=self.res_block2(x)
        x=self.res_block3(x)
        x=self.res_block4(x)
        x=self.avepool(x)
        x=x.view((x.shape[0],-1))
        return self.fc(x)
        
if __name__=='__main__':
    net=resnet18()       
    from torchstat import stat
    stat(net, (3, 32, 32))