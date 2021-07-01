import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Nvidia_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Nvidia_Model=nn.Sequential(
                                   nn.Conv2d(3,24,kernel_size=5,stride=2),
                                   nn.BatchNorm2d(24),
                                   nn.ELU(),

                                   nn.Conv2d(24,36,kernel_size=5,stride=2),
                                   nn.BatchNorm2d(36),
                                   nn.ELU(),

                                   nn.Conv2d(36,48,kernel_size=5,stride=2),
                                   nn.BatchNorm2d(48),
                                   nn.ELU(),

                                   nn.Conv2d(48,64,kernel_size=3,stride=1),
                                   nn.BatchNorm2d(64),
                                   nn.ELU(),

                                   nn.Conv2d(64,64,kernel_size=3,stride=1),
                                   nn.BatchNorm2d(64),
                                   nn.ELU(),

                                   nn.Flatten(),
                                   nn.Linear(64*18,100),
                                   nn.ELU(),
                                   nn.Linear(100,50),
                                   nn.ELU(),
                                   nn.Linear(50,10),
                                   nn.ELU(),
                                   nn.Linear(10,1))

    def forward(self,xb):
        return self.Nvidia_Model(xb)

    def training_step(self,batch):
        images,labels=batch['image'],batch['target']
        labels=labels.reshape(-1,1)
        output=self(images)
        loss=F.mse_loss(output,labels)
        return loss
    def validation_step(self,batch):
        images,labels=batch['image'],batch['target']
        labels=labels.reshape(-1,1)
        output=self(images)
        loss=F.mse_loss(output,labels)
        return {'val_loss': loss.detach()}
    def val_train_epoch_end(self,output):
        val_losses=[x['val_loss'] for x in output]
        val_loss=torch.stack(val_losses).mean().item()
        return {'val_loss':val_loss}
    def epoch_end(self,result,epoch):
        print ("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
        epoch, result['train_loss'], result['val_loss']))







