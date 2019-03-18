import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as Data
import matplotlib 
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')
import matplotlib.pyplot as plt
import numpy as np 
import scipy.io as sio
import random 
import os

N_STATES = 4
N_VALUE = 1

def normalize(input,target):
    input[:,0] = (input[:,0]-np.mean(input[:,0]))/np.std(input[:,0])
    input[:,1] = (input[:,1]-np.mean(input[:,1]))/np.std(input[:,1])
    input[:,2] = (input[:,2]-np.mean(input[:,2]))/np.std(input[:,2])
    input[:,3] = (input[:,3]-np.mean(input[:,3]))/np.std(input[:,3])
    target = (target-np.mean(target))/np.std(target)
    return input,target

class Net(nn.Module):
    
    def __init__(self,):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(N_STATES,50)
        #self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,50)
        #self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(50,N_VALUE)
        #self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        #print(x)
        x = self.fc1(x)
        #print(x)
        x = torch.sigmoid(x)
        #print(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        print(x)
        x = self.out(x)
        #print(x)
        return x

print("-----     Net Constructing     -----")
print(" ")
net = Net().cuda()

print(net)
print(" ")
print("-----     Data Load and Preprocess     -----")
print(" ")


input = np.load('input.npy')
target = np.load('target.npy')
data_size = input.shape[0]
input,target = normalize(input,target)
test_index = random.sample(range(input.shape[0]), 500) 
test_input = input[test_index]
test_target = target[test_index]
train_index = list(set(range(input.shape[0]))-set(test_index))
train_input = input[train_index]
train_target = target[train_index]
train_input = torch.from_numpy(train_input).cuda()
train_target = torch.from_numpy(train_target).cuda()

print("Train data size:",train_input.shape[0])
print("Test data size",test_input.shape[0])
print(" ")
print("-----     Setting and netload     -----")
print(" ")
lr = 0.2
optimizer = torch.optim.Adam(net.parameters(),lr = lr)
loss_func = torch.nn.L1Loss()

for epoch in range(int(1e8)):
    prediction = net(train_input)
    loss = loss_func(prediction,train_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss:",loss.data.cpu().numpy(),"| lr:",lr)
