
##  Mini-Project 1: Residual Network Design

Implementation of ResNet architecture to perform image classification and maximize the accuracy on the CIFAR-10 dataset while keeping the trainable parameteres less than 5M.

### Authors

- Drishti Singh (ds6730@nyu.edu)
- Harshada Sinha (hs4703@nyu.edu)
- Vaishnavi Rajput (vr2229@nyu.edu)

## Execution of the code

This code can be executed on google colab.

First of all import the required python libraries.

```bash
import torch
import torchvision
from torch import nn
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import OneCycleLR
from torchsummary import summary
```
 Define transforms for training and testing data:
```bash

transform = transforms.Compose(
		[transforms.RandomCrop(32, padding=4),
		 transforms.RandomHorizontalFlip(),
		 transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```
Download training and testing data:
```bash
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```
Initialize batch size:
```bash
batch=64
```
Create training and testing dataloaders:
```bash
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=4)
```
Get the class names from the dataset:
```bash
classes = trainset.class_to_idx
```
Define basic block:
```bash
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1,dropout =0.2):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)    
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p = dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```
Define Resnet architecture:
```bash
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,dropout =0.2):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p = dropout)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x)))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```
Create lists for train and test loss and accuracies:
```bash
train_losses = []
train_acc = []
test_losses_l1 = []
test_acc_l1 = []
```
Define function train:
```bash
def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar=tqdm(train_loader)
  correct = 0
  processed = 0
  criterion= nn.CrossEntropyLoss().to(device)
  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    y_pred = model(data)
    loss  = criterion(y_pred, target)
    train_losses.append(loss)
    loss.backward()
    optimizer.step()

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
  train_acc.append(100*correct/processed)
```
Define function test:
```bash
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion= nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses_l1.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc_l1.append(100. * correct / len(test_loader.dataset))
```
Check for the cpu or gpu device for training:
```bash
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```
Create the model by defining hyperparameters:
```bash
def ResNet18():
    return ResNet(BasicBlock, [2,3,2,2])
```

```bash
model=ResNet18().to(device)
```
Display the number of trainable parameters:
```bash
sum(p.numel() for p in model.parameters() if p.requires_grad)
```
Display the summary of the model:
```bash
summary(model, input_size=(3,32,32))
```
Initialize batch size Learning Rate annd moment:
```bash
epochs =50
LR = 0.01
moment = 0.9
```
Train the model over training and testing datasets:
```bash
# optimizer = optim.RMSprop(model.parameters(), lr=LR, momentum =moment)
# optimizer = optim.Adam(model.parameters(),lr= LR,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=moment)
scheduler = OneCycleLR(optimizer,max_lr=0.1,total_steps=epochs)

for epoch in range(epochs):
    print(f'Epoch: {epoch} Learning_Rate {scheduler.get_lr()}')
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
```
Convert from cuda tensors to .cpu()
```bash

train_losses=torch.Tensor(train_losses).cpu()
arr_train=np.array(train_losses)

test_losses_l1=torch.Tensor(test_losses_l1).cpu()
arr_test=np.array(test_losses_l1)

train_acc=torch.Tensor(train_acc).cpu()
arr_train_acc=np.array(train_acc)

test_acc_l1=torch.Tensor(test_acc_l1).cpu()
arr_test_acc=np.array(test_acc_l1)
```
Get the train loss for the last batch for every epoch:
```bash
arr_train1 =[]
for i in range(len(arr_train)):
  if (i%782) == 0:
    arr_train1.append(arr_train[i])
```
Plot the accuracy:
```bash
plt.plot(arr_train_acc)
plt.plot(arr_test_acc)
plt.legend(["train","test"])
plt.title("Epoch vs Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
```
Plot train loss:
```bash
plt.plot(arr_train1 )
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss'])
plt.title('Train Loss vs Epoch')
plt.show()
```
Plot test loss:
```bash
plt.plot(arr_test)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Test loss'])
plt.title('Test Loss vs Epoch')
plt.show()
```
Save the model with all the parameters:
```bash
torch.save(model.state_dict(), 'miniProject1.pt')
```
Perform sanity check:
```bash
test_model = ResNet(BasicBlock, [2,3,2,2])
```
Key Matching:
```bash
test_model.load_state_dict(torch.load('miniProject1.pt'))
```
Place model on GPU:
```bash
test_model.to(device)
```

