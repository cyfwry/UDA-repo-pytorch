import pickle
import glob,os
import random
from PIL import Image
import numpy as np

image_list=glob.glob('.\cifar-10-batches-py\cifar10_test\*')
print(len(image_list))
sum=1000
#read
label_list=[]

for i in range(10):
    label_list+=[[]]
for i in image_list:
    img=Image.open(i)
    img_array=np.asarray(img)
    label_list[int(i[35])]+=[[img_array]]
    
for i in label_list:
    random.shuffle(i)
    
test_dict={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

for i in range(sum):
    for j in range(10):
        test_dict[j]+=label_list[j][i]
        
with open('test.pkl','wb') as test:
    pickle.dump(test_dict,test)

