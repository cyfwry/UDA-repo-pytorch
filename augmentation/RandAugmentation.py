from augmentation import augmentation_transforms,policies
#import augmentation_transforms,policies
import numpy as np
import cv2

means = [0.49139968, 0.48215841, 0.44653091]
stds = [0.24703223, 0.24348513, 0.26158784]

def RandAugmentation(image):
    #img 是numpy格式
    aug_policies = policies.randaug_policies()
    chosen_policy = aug_policies[np.random.choice(len(aug_policies))]
    aug_image = augmentation_transforms.apply_policy(chosen_policy, image)
    aug_image = augmentation_transforms.cutout_numpy(aug_image) 
    return aug_image
    
if __name__=='__main__':
    img=cv2.imread('square.jfif')
    cv2.imshow('img',img)
    t_img=img
    img=(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255-means)/stds
    img_out=RandAugmentation(img)
    print(img_out)
    img_out=(img_out*stds+means)*255
    img_out[img_out>255]=255
    img_out[img_out<0]=0
    img_out=img_out.astype(np.uint8)
    img_out=cv2.cvtColor(img_out,cv2.COLOR_RGB2BGR)
    print(img_out.shape)
    print(img_out-t_img)
    cv2.imshow('img_out',img_out)
    
    cv2.waitKey()