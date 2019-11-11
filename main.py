import argparse, os,glob
import torch
import torch.nn as nn
import numpy as np
from dataset import sup_dataset,uns_dataset
from model_resnet18 import resnet18
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import copy
import time
import cv2
import tqdm
from tensorboardX import SummaryWriter

default_parameter={'channels':64,'batchsize':32,'lr':0.004,'epoch':100,'sup_path':'../set/sup.pkl','uns_path':'../set/uns.pkl','test_path':'../set/test.pkl','threads':16,'ratio':9}

parser=argparse.ArgumentParser(description="UDA")
parser.add_argument("--cuda",action="store_true",help="use cuda")
parser.add_argument("--channels",type=int,default=default_parameter['channels'],help="net channels")
parser.add_argument("--batchsize",type=int,default=default_parameter['batchsize'],help="train batchsize")
parser.add_argument("--lr",type=float,default=default_parameter['lr'],help="learning rate")
parser.add_argument("--epoch",type=int,default=default_parameter['epoch'],help="training epoch")
parser.add_argument("--sup_path",type=str,default=default_parameter['sup_path'],help="supervised data path")
parser.add_argument("--uns_path",type=str,default=default_parameter['uns_path'],help="unsupervised data path")
parser.add_argument("--test_path",type=str,default=default_parameter['test_path'],help="test data path")
parser.add_argument("--model_path",type=str,help="pretrained model path")
parser.add_argument("--threads",type=int,default=default_parameter['threads'],help="multithreads")
#parser.add_argument("--GPUs",type=int,default=default_parameter['GPUs'],help="multiparallel")#这里没进一步写
parser.add_argument("--ratio",type=int,default=default_parameter['ratio'],help="the ratio between unsupervised data and supervised data")

rm_list=glob.glob('../Result/*')
for i in rm_list:
    os.remove(i)
    
writer=SummaryWriter('../Result')

def main():
    global opt,model
    opt=parser.parse_args()
    print(opt)
    
    model_save_path='../model'

    print('===> loading data')
    sup_train_set=sup_dataset(opt.sup_path)
    sup_training_dataloader=DataLoader(dataset=sup_train_set,num_workers=opt.threads,batch_size=opt.batchsize,shuffle=True)
    
    uns_train_set=uns_dataset(opt.uns_path)
    uns_training_dataloader=DataLoader(dataset=uns_train_set,num_workers=opt.threads,batch_size=opt.batchsize*opt.ratio,shuffle=True)
    
    test_training_set=sup_dataset(opt.test_path)
    test_training_dataloader=DataLoader(dataset=test_training_set,num_workers=opt.threads,batch_size=100,shuffle=True)
    print('===> building model')
    labeled_criterion=nn.CrossEntropyLoss()
    unlabeled_criterion=nn.KLDivLoss(reduction='batchmean')#加reduction='sum'后会整体求和（包括batchsize和分类），否则会将求出的KL散度对整体求平均（包括batchsize和分类）,而batchmean只对batch求平均,符合KL的定义,但是batchsize是1时有bug(会对整个结果平均,unsqueezeu也不行)
    model=resnet18()
    model=model.double()#加上，都要double的
    
    if opt.cuda:
        labeled_criterion=labeled_criterion.cuda()
        unlabeled_criterion=unlabeled_criterion.cuda()
        model=model.cuda()        
    #注意，如果想要使用.cuda()方法来将model移到GPU中，一定要确保这一步在构造Optimizer之前。因为调用.cuda()之后，model里面的参数已经不是之前的参数了。
    optimizer=optim.Adam(model.parameters(),lr=opt.lr,momentum=0.9,weight_decay=1e-4)
    start_epoch=0
    if opt.model_path:
        print('===> loading pretrained model')
        pretrained=torch.load(opt.model_path)
        model.load_state_dict(pretrained['model'])
        start_epoch=pretrained['epoch']
    
    print('===> training model')
    model.train()
    torch.backends.cudnn.benchmark = True

    val_best=0
    for epoch in range(start_epoch+1,opt.epoch+1):        
        sup_batch_generator=iter(sup_training_dataloader)
        loss_epoch=train(sup_batch_generator,uns_training_dataloader,labeled_criterion,unlabeled_criterion,model,optimizer,epoch)
        writer.add_scalar('loss', loss_epoch, epoch)
        val_true=0        
        with torch.no_grad():
            for _,batch in enumerate(test_training_dataloader,1):
                input,label=batch[0],batch[1]
                input=input.cuda()
                output=nn.functional.softmax(model(input),-1)
                output=output.cpu()
                output=np.argmax(output.numpy(),-1)
                label=label.numpy()       
                val_true+=(output==label).sum()
        print('loss:{:.4f}'.format(loss_epoch))        
        print('val:{:.2f}'.format(val_true/10000*100))
        writer.add_scalar('val_cor',val_true/10000*100,epoch)
        save_model(model,epoch,model_save_path+'/experiment.pth')
        if val_true/10000>val_best:
            val_best=val_true/10000
            save_model(model,epoch,model_save_path+'/best.pth')
        
def train(sup_batch_generator,uns_training_dataloader,labeled_criterion,unlabeled_criterion,model,optimizer,epoch):
    print("Epoch = {}".format(epoch))
    new_lr=opt.lr/(pow(2,(epoch//30)))
    for i in optimizer.param_groups:
        i['lr']=new_lr
    print('Now Lr is {}'.format(new_lr))
    #processor=tqdm.tqdm(range(len(uns_training_dataloader)))
    processor=tqdm.tqdm(uns_training_dataloader)
    loss_sum=0  
    loss_sup=0
    loss_uns=0  
    train_sup_true=0
    train_uns_true=0
    threshold=(epoch/opt.epoch)*9/10+1/10
    loss=0
    print(threshold)    
    for iteration,batch in enumerate(processor,1):# 看看各输入输出是否requires_grad？拷贝的模型和原模型是否requires_grad分离？

        input,augment,label=batch[0],batch[1],batch[2] #从__getitem__出来时还是numpy，这里却已经成为tensor了？16*32*32*3,requires_grad=False

        if opt.cuda:
            input=input.cuda()
            augment=augment.cuda()
        
        output=model(input)#模型需要B*C*H*W的图片
        output=output.detach()
        out_augment=model(augment)
        output=nn.functional.softmax(output,-1)     
        out_augment_softmax=nn.functional.softmax(out_augment,-1)       
        out_augment=nn.functional.log_softmax(out_augment,-1)

        #这里是加了softmax temperature
        uns_output=torch.tensor([]).double().cuda()
        uns_output_augment=torch.tensor([]).double().cuda()
        for i in range(output.shape[0]):
            if torch.max(output[i]).item()>=threshold and torch.max(out_augment_softmax[i]).item()>=threshold:
                uns_output=torch.cat((uns_output,output[i].unsqueeze(0)))
                uns_output_augment = torch.cat((uns_output_augment, out_augment[i].unsqueeze(0)))

        uns_true=np.argmax(output.detach().cpu().numpy(),-1)
        train_uns_true+=(uns_true==label.numpy()).sum()

        a=0
        if uns_output.shape[0]!=0:
            temp=unlabeled_criterion(uns_output_augment,uns_output)*10 #这里取lambda为10
            a=temp.item()
            loss_uns+=temp.item()
            loss+=temp#作为target的只求softmax，作为output的求log_softmax
        
                
        input,label=next(sup_batch_generator)       #label是一维的      
        
        if opt.cuda:
            input=input.cuda()
            label=label.cuda()
        
        output=model(input) #outpu可以不要grad？          
        softmax_output=nn.functional.softmax(output,-1)
        
        sup_true=np.argmax(softmax_output.detach().cpu().numpy(),-1)
        train_sup_true+=(sup_true==label.cpu().numpy()).sum()

        # temp=labeled_criterion(output.squeeze(),label)#这个是没有TSA的
        # b=temp.item()
        # loss_sup+=temp.item()        
        # loss+=temp
        
        train_output=torch.tensor([]).double().cuda() #torch.tensor([])声明出来的默认是FloatTensor
        train_label=torch.tensor([]).long().cuda()
        for i in range(output.shape[0]):
            if softmax_output[i][label[i]]<threshold:
                train_output=torch.cat((train_output,output[i].unsqueeze(0)))#tensor用[]选取时 会失去被选取的维度 降低一个维度 1维会变成0维
                train_label=torch.cat((train_label,label[i].unsqueeze(0)))

        b=0
        if train_output.shape[0]!=0:            
            temp=labeled_criterion(train_output,train_label) #这个得要
            loss_sup+=temp.item()
            b=temp.item()
            loss+=temp
        
        if isinstance(loss,torch.Tensor):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()

        processor.set_description('loss:{:.4f},loss_uns:{:.4f},loss_sup:{:.4f}'.format(loss,a,b))
        processor.update(1)       
        loss=0    
            
    processor.close()
    loss_sup=loss_sup/(len(uns_training_dataloader))
    print('loss_sup:{:.4f}'.format(loss_sup))
    writer.add_scalar('loss_sup', loss_sup, epoch)
    loss_uns=loss_uns/(len(uns_training_dataloader))
    print('loss_uns:{:.4f}'.format(loss_uns))
    writer.add_scalar('loss_uns', loss_uns, epoch)
    sup_cor=train_sup_true/len(uns_training_dataloader)/opt.batchsize*100
    print('sup_cor:{:.2f}'.format(sup_cor))
    writer.add_scalar('sup_cor', sup_cor, epoch)
    uns_cor=train_uns_true/len(uns_training_dataloader)/opt.batchsize/opt.ratio*100
    print('uns_cor:{:.2f}'.format(uns_cor))
    writer.add_scalar('uns_cor', uns_cor, epoch)
    return loss_sum/(len(uns_training_dataloader))
    
def save_model(model,epoch,path):
    save_dict={'model':model.state_dict(),'epoch':epoch}
    torch.save(save_dict,path)
    print("Epoch {} has been saved in {}.".format(epoch,path))        
    
if __name__=="__main__":
    main()
