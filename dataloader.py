import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import DataLoader,Dataset, random_split
from PIL import Image
import custom_transforms as ct
import matplotlib.pyplot as plt
header_list=['c_image','l_image','r_image','steering_angle','arb_1','arb_2','arb_3']
class SimulatorDataSet(Dataset):
  def __init__(self,csv_file,transform=None):
    self.simulator_data=pd.read_csv(csv_file,names=header_list)
    self.transform=transform
    self.image_paths=np.array(self.simulator_data.iloc[0:len(self.simulator_data),0:3])
    self.target=np.array(self.simulator_data.iloc[0:len(self.simulator_data),3])
  def __len__(self):
    return(len(self.simulator_data))

  def __getitem__(self,idx):
    image_path=self.image_paths[idx]
    image=[Image.open('C:\\Users\\ARYA-PC\\Downloads\\data\\dataset\\dataset\\IMG\\'+image_path[i].split("IMG\\")[1]) for i in range(3)]
    target=self.target[idx]
    sample={'image':image, 'target':target}

    if self.transform:
      sample=self.transform(sample)
    return sample

