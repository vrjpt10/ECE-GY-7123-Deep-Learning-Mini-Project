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

def project1_model():
    return project1_model(BasicBlock, [2,3,2,2])

# Geting cpu or gpu device for training:
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Sanity check:
model = project1_model(BasicBlock, [2,3,2,2])

# Key matching:
model.load_state_dict(torch.load('miniProject1.pt'))


model.eval()

# Key matching:
model.load_state_dict(torch.load('miniProject1.pt'))

# Place model on GPU
model.to(device)

model.eval()

# Initializing batch size:
batch=64

# Defining transforms for training and testing data:
transform = transforms.Compose(
		[transforms.RandomCrop(32, padding=4),
		 transforms.RandomHorizontalFlip(),
		 transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=4)

test_losses_l1 = []
test_acc_l1 = []

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


# Initializing batch size Learning Rate annd moment:
epochs =50
LR = 0.01
moment = 0.9
#added code

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=moment)
scheduler = OneCycleLR(optimizer,max_lr=0.1,total_steps=epochs)

for epoch in range(epochs):
    print(f'Epoch: {epoch} Learning_Rate {scheduler.get_lr()}')
    # train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()


test_losses_l1=torch.Tensor(test_losses_l1).cpu()
arr_test=np.array(test_losses_l1)

test_acc_l1=torch.Tensor(test_acc_l1).cpu()
arr_test_acc=np.array(test_acc_l1)

# Getting the train loss for the last batch for every epoch:
# arr_train1 =[]
# for i in range(len(arr_train)):
#   if (i%782) == 0:
#     arr_train1.append(arr_train[i])

# Plotting the accuracy:
# plt.plot(arr_train_acc)
plt.plot(arr_test_acc)
plt.legend(["train","test"])
plt.title("Epoch vs Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# Printing train loss:
# plt.plot(arr_train1 )
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss'])
plt.title('Train Loss vs Epoch')
plt.show()

# Printing test loss:
plt.plot(arr_test)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Test loss'])
plt.title('Test Loss vs Epoch')
plt.show()

#  Save random state of the model:
# torch.save(model.state_dict(), 'miniProject1.pt')

