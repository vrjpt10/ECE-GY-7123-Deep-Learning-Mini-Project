# Importing libraries:
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
from model import ResNet

# Defining transforms for training and testing data:
transform = transforms.Compose(
		[transforms.RandomCrop(32, padding=4),
		 transforms.RandomHorizontalFlip(),
		 transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Downloading training and testing data:
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


# Initializing batch size:
batch=64

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=4)

# Getting class names from the dataset:
classes = trainset.class_to_idx

# Creating lists for train and test loss and accuracies:
train_losses = []
train_acc = []
test_losses_l1 = []
test_acc_l1 = []

# Function train:
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

# Geting cpu or gpu device for training:
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#added code
def project1_model():
    return project1_model(BasicBlock, [2,3,2,2])
# Loading model:
model=project1_model().to(device)

# Displaying number of trainable parameters:
sum(p.numel() for p in model.parameters() if p.requires_grad)

# Displaying summary of the model:
summary(model, input_size=(3,32,32))

# Initializing batch size Learning Rate annd moment:
epochs =50
LR = 0.01
moment = 0.9

# Training the model over training and testing datasets:

# optimizer = optim.RMSprop(model.parameters(), lr=LR, momentum =moment)
# optimizer = optim.Adam(model.parameters(),lr= LR,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=moment)
scheduler = OneCycleLR(optimizer,max_lr=0.1,total_steps=epochs)

for epoch in range(epochs):
    print(f'Epoch: {epoch} Learning_Rate {scheduler.get_lr()}')
    train(model, device, train_loader, optimizer, epoch)
    # test(model, device, test_loader)
    scheduler.step()

#converting from cuda tensors to .cpu()
train_losses=torch.Tensor(train_losses).cpu()
arr_train=np.array(train_losses)

# test_losses_l1=torch.Tensor(test_losses_l1).cpu()
# arr_test=np.array(test_losses_l1)

train_acc=torch.Tensor(train_acc).cpu()
arr_train_acc=np.array(train_acc)

# test_acc_l1=torch.Tensor(test_acc_l1).cpu()
# arr_test_acc=np.array(test_acc_l1)

# Getting the train loss for the last batch for every epoch:
arr_train1 =[]
for i in range(len(arr_train)):
  if (i%782) == 0:
    arr_train1.append(arr_train[i])

# Plotting the accuracy:
plt.plot(arr_train_acc)
# plt.plot(arr_test_acc)
plt.legend(["train","test"])
plt.title("Epoch vs Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Printing train loss:
plt.plot(arr_train1 )
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss'])
plt.title('Train Loss vs Epoch')
plt.show()

# Printing test loss:
# plt.plot(arr_test)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Test loss'])
plt.title('Test Loss vs Epoch')
plt.show()

# Save random initial weights:
torch.save(model.state_dict(), 'miniProject1.pt')
# Sanity check:
model = ResNet(BasicBlock, [2,3,2,2])

# Key matching:
model.load_state_dict(torch.load('miniProject1.pt'))

# Place model on GPU
model.to(device)

model.eval()

