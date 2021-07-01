import random
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import DataLoader,Dataset, random_split, TensorDataset
from PIL import Image
import custom_transforms as ct
import matplotlib.pyplot as plt
header_list=['c_image','l_image','r_image','steering_angle','arb_1','arb_2','arb_3']
class SimulatorDataSet():
  def __init__(self,csv_file,transform=None):
    self.simulator_data=pd.read_csv(csv_file,names=header_list)
    self.transform=transform
    self.image_paths=np.array(self.simulator_data.iloc[0:len(self.simulator_data),0:3])
    self.target=np.array(self.simulator_data.iloc[0:len(self.simulator_data),3])
    self.images=[]
    self.targets=[]
    for n in range(len(self.simulator_data)):
        self.get_image(n)
    #return {'images':self.images, 'targets':self.targets}
  def __len__(self):
    return(len(self.simulator_data))

  def get_image(self,idx):
    image_path=self.image_paths[idx]
    img_choose=random.randint(0,2)
    image=Image.open('C:\\Users\\ARYA-PC\\Downloads\\data\\dataset\\dataset\\IMG\\'+image_path[img_choose].split("IMG\\")[1])
    if img_choose==0:
        target=self.target[idx]
    elif img_choose==1:
        target=self.target[idx]+0.2
    else:
        target=self.target[idx]-0.2

    sample={'image':image, 'target':target}

    if self.transform:
      sample=self.transform(sample)
    self.images.append(sample['image'])
    image.close()
    self.targets.append(sample['target'])

if __name__ == '__main__':
  trans=tt.Compose([ct.MiddleCrop(size=(80,320)),ct.RandomHorizontalFlip(0.5),ct.ChangeBright(0.4,0.4,0,0),ct.Resize([66,200]),ct.ConvertTensor()])
  dataset=SimulatorDataSet(csv_file='C:\\Users\\ARYA-PC\\Downloads\\data\\dataset\\dataset\\driving_log.csv',transform=trans)

  images,targets=dataset.images, dataset.targets

  print(images[0],targets[0])
  data=TensorDataset(images,targets)

  train_number = 0.8 * len(dataset)
  val_number=len(dataset)-train_number

  train_data,test_data=random_split(data,[train_number,val_number])
