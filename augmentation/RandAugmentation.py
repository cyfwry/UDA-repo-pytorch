from augmentation import augmentation_transforms,policies
import numpy as np

def RandAugmentation(image):
    #img 是numpy格式
    aug_policies = policies.randaug_policies()
    chosen_policy = aug_policies[np.random.choice(len(aug_policies))]
    aug_image = augmentation_transforms.apply_policy(chosen_policy, image)
    aug_image = augmentation_transforms.cutout_numpy(aug_image) #?
    return aug_image
    
#if __name__=='__main__':
#    RandAugmentation(np.random.random((100,100,3)))