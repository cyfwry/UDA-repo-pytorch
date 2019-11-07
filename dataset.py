import pickle
import torch.utils.data as data
import torch
import numpy as np
import random
class UDAdataset(data.Dataset):#32*32*3->3*32*32,transpose?
    def __init__(self,path):#改写为混编
        super(UDAdataset,self).__init__()
        with open(path,'rb') as file:
            dataset=pickle.load(file)
            
        self.data=[]
        self.label=[]       
        for i in dataset.keys():
            for j in range(len(dataset[0])):
                self.data+=[dataset[i][j].astype(np.float64).transpose((2,0,1))/255]
                self.label+=[np.array(i,dtype=np.int64)]
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)