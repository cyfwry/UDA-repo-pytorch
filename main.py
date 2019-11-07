import argparse, os
import torch
import torch.nn as nn
import numpy as np
from dataset import UDAdataset
from model import UDA
from torch.utils.data import DataLoader
from augmentation import RandAugmentation as RA
import torch.optim as optim
import random
import copy

default_parameter={'channels':16,'batchsize':16,'lr':0.1,'epoch':50,'sup_path':'sup.pkl','uns_path':'uns.pkl','model':'','save_path':'model','threads':1,'ratio':4}

parser=argparse.ArgumentParser(description="UDA")
parser.add_argument("--cuda",action="store_true",help="use cuda")
parser.add_argument("--channels",type=int,default=default_parameter['channels'],help="net channels")
parser.add_argument("--batchsize",type=int,default=default_parameter['batchsize'],help="train batchsize")
parser.add_argument("--lr",type=float,default=default_parameter['lr'],help="learning rate")
parser.add_argument("--epoch",type=int,default=default_parameter['epoch'],help="training epoch")
parser.add_argument("--sup_path",type=str,default=default_parameter['sup_path'],help="supervised data path")
parser.add_argument("--uns_path",type=str,default=default_parameter['uns_path'],help="unsupervised data path")
parser.add_argument("--model",type=str,default=default_parameter['model'],help="pretrained model path")
parser.add_argument("--save_path",type=str,default=default_parameter['save_path'],help="save path")
parser.add_argument("--threads",type=int,default=default_parameter['threads'],help="multithreads")
#parser.add_argument("--GPUs",type=int,default=default_parameter['GPUs'],help="multiparallel")#这里没进一步写
parser.add_argument("--ratio",type=int,default=default_parameter['ratio'],help="the ratio between unsupervised data and supervised data")
#改一下打包的ratio 让它们统一定义
def main():
    global opt,model
    opt=parser.parse_args()
    print(opt)
    
    print('===> loading data')
    sup_train_set=UDAdataset(opt.sup_path)
    sup_training_dataloader=DataLoader(dataset=sup_train_set,num_workers=opt.threads,batch_size=opt.batchsize,shuffle=True)
    uns_train_set=UDAdataset(opt.uns_path)
    uns_training_dataloader=DataLoader(dataset=uns_train_set,num_workers=opt.threads,batch_size=opt.batchsize,shuffle=True)
    
    print('===> building model')
    labeled_criterion=nn.CrossEntropyLoss()
    unlabeled_criterion=nn.KLDivLoss(reduction='sum')#要加reduction='sum'，否则会将求出的KL散度对类数求平均
    model=UDA(64)
    model=model.double()#加上，都要double的
    
    if opt.cuda:
        labeled_criterion=labeled_criterion.cuda()
        unlabeled_criterion=unlabeled_criterion.cuda()
        model=model.cuda()        
    #注意，如果想要使用.cuda()方法来将model移到GPU中，一定要确保这一步在构造Optimizer之前。因为调用.cuda()之后，model里面的参数已经不是之前的参数了。
    optimizer=optim.SGD(model.parameters(),lr=opt.lr,momentum=0.9,weight_decay=1e-4)
    
    if opt.model:
        print('===> loading pretrained model')
        weights=torch.load(opt.model)
        model.load_state_dict(weights.state_dict())
    
    print('===> training model')
    model.train()
    torch.backends.cudnn.benchmark = True
    for epoch in range(opt.epoch):
        train(sup_training_dataloader,uns_training_dataloader,labeled_criterion,unlabeled_criterion,model,optimizer,epoch)
        #save_model(model,epoch)

        
def train(sup_training_dataloader,uns_training_dataloader,labeled_criterion,unlabeled_criterion,model,optimizer,epoch):
    print("Epoch = {}".format(epoch))
    
    threshold=((epoch+1)/opt.epoch)*8/10+2/10
    loss=0
    cor_num=0
    model_copy=copy.deepcopy(model)
    for param in model_copy.parameters():
        param.requires_grad=False 
        
    for iteration,batch in enumerate(uns_training_dataloader,1):# 看看各输入输出是否requires_grad？拷贝的模型和原模型是否requires_grad分离？
        '''
        input,label=batch[0],batch[1] #从__getitem__出来时还是numpy，这里却已经成为tensor了？16*32*32*3,requires_grad=False

        augment=[]
        for i in input:
            augment+=[RA.RandAugmentation(i.numpy().transpose((1,2,0))).transpose((2,0,1))]
        
        augment=torch.tensor(augment)
        
        if opt.cuda:
            input=input.cuda()
            augment=augment.cuda()
        
        output=model_copy(input)#模型需要B*C*H*W的图片
        out_augment=model(augment)
        output=nn.functional.softmax(output,-1)
        out_augment=nn.functional.log_softmax(out_augment,-1)
        
        loss+=unlabeled_criterion(out_augment,output)#作为target的只求softmax，作为output的求log_softmax
        
        for i in range(opt.batchsize):
            pred_label=0
            max=0
            for j in range(10):
                if max<output[i][0][j]:
                    max=output[i][0][j]
                    pred_label=j
            if(label[i]==pred_label):
                cor_num+=1        
        
        if iteration%(10*opt.ratio) == 0:
            print('Epoch : {},Iteration : {}/{},Loss : {},Accuracy : {}'.format(epoch,iteration,len(uns_training_dataloader),loss.item(),cor_num/(opt.batchsize*opt.ratio*10)))
            cor_num=0
        '''    
        def cycle():
            while(True):
                for _,generator in enumerate(sup_training_dataloader):
                    yield generator
        
        if iteration%opt.ratio==0:
            sup_batch_generator=cycle()#这个和前面的batch不一样，那个是list，这个是generator
            input,label=next(sup_batch_generator)       #label就是16的大小，一维    
            
            if opt.cuda:
                input=input.cuda()
                label=label.cuda()
            
            output=model(input) #output可不可以不要grad？  16*1*10              
            softmax_output=nn.functional.softmax(output,-1)
            
            loss=labeled_criterion(output.squeeze(),label)            
            for i in model.parameters():
                print(i)
            print(loss)
            '''
            train_output=torch.tensor([]).double() #torch.tensor([])声明出来的默认是FloatTensor
            train_label=torch.tensor([]).long()
            for i in range(output.shape[0]):
                if (softmax_output[i]<threshold).all():
                    train_output=torch.cat((train_output,output[i]))#tensor用[]选取时 会失去被选取的维度 降低一个维度 1维会变成0维
                    train_label=torch.cat((train_label,label[i].unsqueeze(-1)))
            
            print(train_output.shape)#这边的loss是否正常，还没有查看
            #这边的cor_num还没有统计
            before=loss.item()
            if train_output.shape[0]!=0:               
                loss+=labeled_criterion(train_output,train_label)
            print('--------------------------------------------------------------------------')
            print(nn.functional.softmax(train_output,-1))
            print('**************************************************************************')
            print(train_label)
            print('??????????????????????????????????????????????????????????????????????????')
            print(labeled_criterion(train_output,train_label))
            print(before)
            print(loss)
            '''
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss=0    
            model_copy=copy.deepcopy(model)
            for param in model_copy.parameters():
                param.requires_grad=False
                   
def save_model(model,epoch):
    torch.save(model.state_dict(),opt.save_path+'/'+str(epoch)+'.pth')
    print("Epoch {} has been saved.".format(epoch))    
    
if __name__=="__main__":
    main()
