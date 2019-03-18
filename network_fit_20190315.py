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

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden4 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden5 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden6 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden7 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden8 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden9 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden12 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden13 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden14 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden15 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden16 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden17 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden18 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden19 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        x = F.relu(self.hidden12(x))
        x = F.relu(self.hidden13(x))
        x = F.relu(self.hidden14(x))
        x = F.relu(self.hidden15(x))
        x = F.relu(self.hidden16(x))
        x = F.relu(self.hidden17(x))
        x = F.relu(self.hidden18(x))
        x = F.relu(self.hidden19(x))
        x = self.predict(x)             # linear output
        #print(x)
        return x

print("-----     Net Constructing     -----")
print(" ")
net = Net(n_feature=4, n_hidden=5000, n_output=1).cuda()

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
    train_index = random.sample(range(train_input.shape[0]), 16) 
    input_tmp = train_input[train_index]
    target_tmp = train_target[train_index]
    prediction = net(input_tmp)
    loss = loss_func(prediction,target_tmp)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("loss:",loss.data.cpu().numpy(),"| lr:",lr)
