import random
from PIL import Image
import torchvision.transforms as tt
import numpy as np
import torch
import torchvision
class RandomChoose(object):
    def __init__(self,choice=2):
        self.choice=choice #number of choice left right or center
    def __call__(self,sample):
        image, target=sample['image'], sample['target']
        i=random.randint(0,self.choice)
        if i==0:
            image=image[0]
        elif i==1:
            image=image[1]
            target+=0.2
        else:
            image=image[2]
            target-=0.2
        return {'image':image, 'target':target}

class MiddleCrop(object):
    def __init__(self,size):
        self.size=size
    def __call__(self,sample):
        image, target=sample['image'],sample['target']
        h,w=self.size[0],self.size[1]
        t=tt.CenterCrop((h,w))
        image=t(image)
        return {'image':image, 'target':target}

class RandomHorizontalFlip(object):
    def __init__(self,p=0.5):
        self.p=p
    def __call__(self,sample):
        image, target=sample['image'], sample['target']
        if random.random()>self.p:
            trans=tt.RandomHorizontalFlip(1)
            image=trans(image)
            if target!=0:
                target=-1*target
        return {'image':image,'target':target}

class ConvertTensor(object):
    def __call__(self,sample):
        image, target=sample['image'], sample['target']
        trans=tt.ToTensor()
        image=trans(image)
        target=torch.tensor(float(target))
        return {'image':image, 'target':target}

class ChangeBright(object):
    def __init__(self,brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness=brightness
        self.contrast=contrast
        self.saturation=saturation
        self.hue=hue
    def __call__(self,sample):
        image, target=sample['image'],sample['target']
        trans=tt.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        image=trans(image)
        return {'image':image,'target':target}

class Resize(object):
    def __init__(self,input_size):
        self.h=input_size[0]
        self.w=input_size[1]
    def __call__(self,sample):
        image, target=sample['image'], sample['target']
        trans=tt.Resize(size=(self.h,self.w))
        image=trans(image)
        return {'image':image,'target':target}

class Normalize(object):
    def __init__(self,mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)):
        self.mean=mean
        self.std=std
    def __call__(self,sample):
        image, target=sample['image'], sample['target']
        trans=tt.Normalize(mean=self.mean,std=self.std)
        image=trans(image)
        return {'image':image, 'target':target}