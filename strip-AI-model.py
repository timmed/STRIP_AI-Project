import os
from zipfile import ZipFile
import shutil
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import GPUtil
import gc
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()
print(nvsmi.DeviceQuery('memory.free, memory.total'))
#print(nvsmi.DeviceQuery('--help-query-gpu'), end='\n')

# from torchsummary import summary

import matplotlib.pyplot as plt
import matplotlib

# plt.ion()
matplotlib.use("TkAgg")
from IPython.display import display

# display.set_matplotlib_formats('svg')


GPU = GPUtil.getAvailable()
use_cuda = torch.cuda.is_available()
print(use_cuda)
print(GPU)


device = torch.device('cuda' if use_cuda else 'cpu')
#print("this is how much is on GPU", torch.cuda.list_gpu_processes(device))



loaded_images = []
labels = []
for folder in os.listdir(os.getcwd() + '\STRIP_AI\\processed_data'):
    #print(folder)
    for file in os.listdir(os.getcwd() + '\STRIP_AI\\processed_data' + '\\' + folder):
        labels.append(folder)
        #print(file)
        image = cv2.imread(os.getcwd() + '\STRIP_AI\\processed_data'  + '\\' + folder + '\\' + file)
        image_array = np.array(image)
        loaded_images.append(image_array)


# print(labels)
# print(len(labels))
print(loaded_images)
print(len(loaded_images))

"""Converting Labels"""
# labels = np.array(labels)
# l_encoder = LabelEncoder()
# labels = l_encoder.fit_transform(labels)
coded_labels = []
for i in range(len(loaded_images)):
    if labels[i] == 'CE':
        coded_labels.append(0)
    else:
        coded_labels.append(1)
'''For labels'''

# labels = [[value for a in range(375)] for value in range(2)]
# labels = np.array(labels).reshape(-1)
#print(labels)

l = 500
w = 500

resized_images = []
#counter2 = 0
for image in loaded_images:
    image = cv2.resize(image, dsize=(l, w), interpolation=cv2.INTER_NEAREST)
    resized_images.append(image)
    # counter2 += 1
    # print(counter2)

# print(resized_images[3].shape)
# print(resized_images)

data = np.array(resized_images)
data = data.astype(float)

print("ok. Getting ready to print labels")
transform = T.Compose([T.ToTensor()])

dataT = torch.tensor(data).float()
dataT = dataT.to(torch.float64)
print(type(dataT))
labelsT = torch.tensor(coded_labels).long()


transformed_Image = []
img_Norm = []
counter = 0
for image in range(len(data)):
    transformed_trainData = transform(data[image, :, :, :])
    mean, std = transformed_trainData.mean([1, 2]), transformed_trainData.std([1, 2])
    #print(std)
    transform_norm = T.Compose([T.Normalize(mean, std)])
    #transform_norm = transform_norm.to(torch.float64)
    img_normalized = transform_norm(transformed_trainData)
    counter += 1
    #print(counter)
    transformed_Image.append(transformed_trainData)
    img_Norm.append(img_normalized)

img_Norm = torch.stack(img_Norm).float()

# Use scikitlearn to split the data
train_data, test_data, train_labels, test_labels = train_test_split(img_Norm, labelsT, test_size=.1)

# Step 3: convert into PyTorch Datasets
train_data = TensorDataset(train_data, train_labels)
test_data = TensorDataset(test_data, test_labels)

batchsize = 10
train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

'''Create the Model'''


def makeTheNet(printtoggle=False):
    class cnn(nn.Module):
        def __init__(self, printtoggle):
            super().__init__()

            self.print = printtoggle

            ### convolution layers ###
            ## Layer 1
            self.conv1 = nn.Conv2d(3, 64, 5, stride=1, padding=2)
            self.pool1 = nn.MaxPool2d(2)
            self.A1 = nn.LeakyReLU(0.1)
            self.bnorm1 = nn.BatchNorm2d(64)
            size1 = np.floor(((500 + (2 * 2) - 5) / 1 + 1) / 2)

            ## Layer 2
            self.conv2 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
            self.pool2 = nn.MaxPool2d(2)
            self.A2 = nn.LeakyReLU(0.1)
            self.bnorm2 = nn.BatchNorm2d(128)
            size2 = np.floor(((size1 + (2 * 2) - 5) / 1 + 1) / 2)

            ## Layer 3
            self.conv3 = nn.Conv2d(128, 256, 5, stride=1, padding=2)
            self.pool3 = nn.MaxPool2d(2)
            self.A3 = nn.LeakyReLU(0.1)
            self.bnorm3 = nn.BatchNorm2d(256)
            size3 = np.floor(((size2 + (2 * 2) - 5) / 1 + 1) / 2)

            ## Layer 4
            self.conv4 = nn.Conv2d(256, 512, 5, stride=1, padding=2)
            self.pool4 = nn.MaxPool2d(2)
            self.A4 = nn.LeakyReLU(0.1)
            self.bnorm4 = nn.BatchNorm2d(512)
            size4 = np.floor(((size3 + (2 * 2) - 5) / 1 + 1) / 2)

            expectedSize = np.floor((size4 + 2 * 0 - 1) / 1) + 1
            expectedSize = 512 * int(expectedSize ** 2)

            ## Fully connected layer
            self.fcl = nn.Linear(expectedSize, 100)
            self.A5 = nn.ReLU()
            self.out = nn.Linear(100, 2)

        ##  Forward pass
        def forward(self, x):
            print(f'Input: {x.shape}') if self.print else None

            # Block 1
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.A1(x)
            x = self.bnorm1(x)

            # Block 2
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.A2(x)
            x = self.bnorm2(x)

            # Block 3
            x = self.conv3(x)
            x = self.pool3(x)
            x = self.A3(x)
            x = self.bnorm3(x)

            # Block 4
            x = self.conv4(x)
            x = self.pool4(x)
            x = self.A4(x)
            x = self.bnorm4(x)

            # flattening layer
            nUnits = x.shape.numel() / x.shape[0]
            x = x.view(-1, int(nUnits))

            x = self.fcl(x)
            x = self.out(x)

            return x

    net = cnn(printtoggle)

    lossFunction = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

    return net, lossFunction, optimizer


def trainerFunction():
    numepochs = 100

    # create  a new model

    net, lossfun, optimizer = makeTheNet()
    net.to(device)
    lossfun.to(device)

    # initialize losses
    losses = torch.zeros(numepochs)
    trainAcc = []
    testAcc = []

    # loop over epochs
    for epochi in range(numepochs):

        # loop over training data batches
        net.train()
        batchAcc = []
        batchLoss = []

        for X, y in train_loader:
            # push data to GPU
            X = X.to(device)
            y = y.to(device)

            # forward pass and loss
            yHat = net(X)
            loss = lossfun(yHat, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())

            # compute accuracy
            matches = torch.argmax(yHat.cpu(), dim=1) == y.cpu()  # booleans (false/true)
            matchesNumeric = matches.float()  # convert to numbers (0/1)
            accuracyPct = 100 * torch.mean(matchesNumeric)  # average and x100
            batchAcc.append(accuracyPct)  # add to list of accuracies
        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        trainAcc.append(np.mean(batchAcc))

        # and get average losses across the batches
        losses[epochi] = np.mean(batchLoss)

        # test accuracy
        net.eval()
        X, y = next(iter(test_loader))  # extract X,y from test dataloader
        X = X.to(device)
        y = y.to(device)
        with torch.no_grad():  # deactivates autograd
            yHat = net(X)

        # compare the following really long line of code to the training accuracy lines
        testAcc.append(100 * torch.mean((torch.argmax(yHat.cpu(), dim=1) == y.cpu()).float()))

    # end epochs

    # function output
    return trainAcc, testAcc, losses, net


trainAcc, testAcc, losses, net = trainerFunction()
print(testAcc)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].plot(losses, 's-')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Model loss')

ax[1].plot(trainAcc, 's-', label='Train')
ax[1].plot(testAcc, 'o-', label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final model test accuracy: {testAcc[-1]:.2f}%')
ax[1].legend()

plt.show()
