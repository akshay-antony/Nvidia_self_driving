import dataloader as dl
import nvidia_model as nm
import custom_transforms as ct
import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import Dataset,DataLoader

def fit(model,train_data,epochs,lr,opt_func=torch.optim.Adam):
    optimizer=opt_func(model.parameters(),lr)
    history=[]
    for i in range(epochs):
        batch_number=0
        train_losses = []
        val_losses = []
        for batch in train_data:
            batch_number+=1
            if batch_number<200:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                val_losses.append(model.validation_step(batch))
        results=model.val_train_epoch_end(val_losses)
        results['train_loss']=torch.stack(train_losses).mean().item()
        model.epoch_end(results,i)
        history.append(results)
    return history

if __name__ == '__main__':
    model=nm.Nvidia_Model()
    trans = tt.Compose([ct.RandomChoose(2), ct.MiddleCrop(size=(80, 320)), ct.RandomHorizontalFlip(0.5),
                        ct.ChangeBright(0.4, 0.4, 0, 0), ct.Resize([66, 200]), ct.ConvertTensor()])
    dataset = dl.SimulatorDataSet(csv_file='C:\\Users\\ARYA-PC\\Downloads\\data\\dataset\\dataset\\driving_log.csv',
                               transform=trans)
    train_data = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    epochs=20
    lr=.0001
    history=fit(model,train_data,epochs,lr)
    torch.save(model.state_dict(),'C:\\Users\\ARYA-PC\\Downloads\\data\\torch_value\\state.pth')
