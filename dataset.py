import pickle
import torch.utils.data as data
import torch
import numpy as np
import random
from augmentation import RandAugmentation as RA

means = [0.49139968, 0.48215841, 0.44653091]
stds = [0.24703223, 0.24348513, 0.26158784]

class sup_dataset(data.Dataset):#32*32*3->3*32*32,transpose?
    def __init__(self,path):#改写为混编
        super(sup_dataset,self).__init__()
        with open(path,'rb') as file:
            dataset=pickle.load(file)
            
        self.data=[]
        self.label=[]       
        for i in dataset.keys():
            for j in range(len(dataset[0])):
                self.data+=[((dataset[i][j]/255-means)/stds).transpose((2,0,1))]
                self.label+=[np.array(i,dtype=np.int64)]
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)

class uns_dataset(data.Dataset):#32*32*3->3*32*32,transpose?
    def __init__(self,path):#改写为混编
        super(uns_dataset,self).__init__()
        with open(path,'rb') as file:
            dataset=pickle.load(file)
            
        self.data=[]     
        self.label=[]  
        for i in dataset.keys():
            for j in range(len(dataset[0])):
                self.data+=[(dataset[i][j]/255-means)/stds]
                self.label+=[np.array(i,dtype=np.int64)]
    def __getitem__(self,index):
        return self.data[index].transpose((2,0,1)),RA.RandAugmentation(self.data[index]).transpose((2,0,1)),self.label[index]
    def __len__(self):
        return len(self.data)
