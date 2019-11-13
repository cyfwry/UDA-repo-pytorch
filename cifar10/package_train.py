import pickle
import glob,os
import random
from PIL import Image
import numpy as np
address=r'./cifar-10-batches-py/cifar10'
image_list=glob.glob('./cifar-10-batches-py/cifar10_train/*')
sup=500
sum=5000
print(len(image_list))
#read
label_list=[]
for i in range(10):
    label_list+=[[]]
for i in image_list:
    img=Image.open(i)
    img_array=np.asarray(img)
    label_list[int(i[36])]+=[[img_array]]
    
for i in label_list:
    random.shuffle(i)
    
sup_dict={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
uns_dict={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for i in range(sup):
    for j in range(10):
        sup_dict[j]+=label_list[j][i]
        
for i in range(sup,sum):
    for j in range(10):
        uns_dict[j]+=label_list[j][i]
        
with open('uns.pkl','wb') as uns:
    pickle.dump(uns_dict,uns)
with open('sup.pkl','wb') as sup:
    pickle.dump(sup_dict,sup)
