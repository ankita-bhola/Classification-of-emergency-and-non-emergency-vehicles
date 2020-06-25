import torch
import torchvision 
from torchvision import models,transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

transform=transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
class loader(Dataset):
    def __init__(self,csv_file,transform):
        self.data=pd.read_csv(csv_file)
        self.transform=transform
    def __len__(self):
        
        return len(self.data)
    def __getitem__(self,idx):
        img_path=self.data.iloc[:,0]
        image=Image.open('train_SOaYf6m/images/'+ img_path[idx])
        if self.transform:
            image=self.transform(image)
        info=self.data.iloc[:,1]
        label=torch.as_tensor(info[idx],dtype=int)
        return(image,label)
trainset=loader('train_SOaYf6m/train.csv',transform=transform)
trainloader=DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
testset=loader('train_SOaYf6m/test.csv',transform=transform)
testloader=DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
model=models.resnet18(pretrained= True)
model.eval()
criterion = nn.CrossEntropyLoss()

path='./car_final.pth'
PATH='./car.pth'

for i in range (2):
    if(i==0):
        for param in model.parameters():
            param.required_grad=False
        model.fc=nn.Linear(512,2)
        optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
        bound=1/math.sqrt(model.fc.weight.size(1))
        model.fc.weight.data.uniform_(-bound,bound)
        model.fc.bias.data.uniform_(-bound,bound)
    else:
        model.fc=nn.Linear(512,2)
        optimizer=optim.SGD(model.parameters(),lr=0.000001,momentum=0.9)
        model.load_state_dict(torch.load(PATH))

    for epoch in range(2):
        running_loss=0
        batch_loss=0
        for i,data in enumerate(trainloader,0):
            images,labels=data
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            print('heloo')
if(i==0):
    torch.save(model.state_dict(),PATH)
else:
    torch.save(model.state_dict(),path)

model.load_state_dict(torch.load(path))

correct=0
total=0
with torch.no_grad():
    for data in testloader:
        images,labels= data
        outputs = model(images)
        _,predicted=torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network : %d %%' % (100 * correct / total))

class test(Dataset):
    def __init__(self,csv_file,transform):
        self.data=pd.read_csv(csv_file)
        self.transform=transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        img_path=self.data.iloc[:,0]
        image=Image.open('train_SOaYf6m/images/'+ img_path[idx])
        imgpath=img_path[idx]
        if self.transform:
            image=self.transform(image)
        return(image,imgpath)

dataset=test('test_vc2kHdQ.csv',transform=transform)
dataloader=DataLoader(dataset,batch_size=2,shuffle=False,num_workers=2)
x=[]
y=[]
for i,data in enumerate(dataloader):
    images,imgpath=data
    outputs=model(images)
    _,predicted=torch.max(outputs.data,1)
    for j in range(2):
        x.append(imgpath[j])
        y.append(predicted[j])


data=np.vstack((x,y)).T
columns=[]
columns=['image_names','emergency_or_not']
dataFrame=pd.DataFrame(data)
dataFrame.columns=columns
dataFrame.to_csv(r'/home/ankita/Desktop/imageclassification/emergency_or_not/resultant.csv')



