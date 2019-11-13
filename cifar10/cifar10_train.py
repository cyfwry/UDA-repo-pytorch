import pickle
import gzip
import glob
import os
import numpy as np
from PIL import Image
list=glob.glob(r'.\cifar-10-batches-py\data_batch_*')
for address in list:
    with open(address,'rb') as file:
        dict=pickle.load(file,encoding='bytes')
        for i in range(len(dict[b'data'])):
            data=np.zeros([32,32,3])
            k=0
            data[:,:,0]=dict[b'data'][i][k:k+1024].reshape((32,32))
            k=k+1024
            data[:,:,1]=dict[b'data'][i][k:k+1024].reshape((32,32))
            k=k+1024
            data[:,:,2]=dict[b'data'][i][k:k+1024].reshape((32,32))
            k=k+1024               
            data=data.astype(np.uint8)
            image=Image.fromarray(data)
            image.save(r'cifar-10-batches-py/cifar10_train/'+str(dict[b'labels'][i])+str(dict[b'filenames'][i],encoding='utf-8'))
'''            
b'batch_label'
b'labels'
b'data'
b'filenames'
'''