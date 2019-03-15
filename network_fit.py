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
#range, range rate, v, a
N_VALUE = 1



def normalize_value(value,lb,ub):
    return 2*(value-lb)/(ub-lb)-1

def normalize(input,target,range_lb,range_ub,range_rate_lb,range_rate_ub,v_lb,v_ub,target_lb,target_ub):
    input[:,0] = (input[:,0]-np.mean(input[:,0]))/np.std(input[:,0])
    input[:,1] = (input[:,1]-np.mean(input[:,1]))/np.std(input[:,1])
    input[:,2] = (input[:,2]-np.mean(input[:,2]))/np.std(input[:,2])
    input[:,3] = (input[:,3]-np.mean(input[:,3]))/np.std(input[:,3])
    target = (target-np.mean(target))/np.std(target)
    return input,target


def apply_test(test_data,test_target):
    whole_loss = 0
    randint = np.random.randint(0,500)
    mydata = test_data[randint]
    mytarget = test_target[randint]
    myprediction = net(mydata)
    #print("prediction:",myprediction,"target:",mytarget)
    prediction = net(test_data)
    #print(test_target)
    #print(prediction)
    loss = loss_func(prediction,test_target)
    return loss

def input_process(input):
    input_size = input.shape[0]
    range_lb = input[:,0].min()
    range_ub = input[:,0].max()
    range_rate_lb = input[:,1].min()
    range_rate_ub = input[:,1].max()
    v_lb = input[:,2].min()
    v_ub = input[:,2].max()
    final_input_size = input_size*31
    new_input = np.zeros((final_input_size,4))
    for i in range(new_input.shape[0]):
        old_input_index = i//31
        old_input_a_number = i%31
        new_input[i][0] = input[old_input_index][0]
        new_input[i][1] = input[old_input_index][1]
        new_input[i][2] = input[old_input_index][2]
        new_input[i][3] = old_input_a_number
    return new_input,range_lb,range_ub,range_rate_lb,range_rate_ub,v_lb,v_ub

def target_process(target):
    target_size = target.shape[0]
    final_target_size = target_size*31
    new_target = np.zeros((final_target_size,1))
    for i in range(new_target.shape[0]):
        old_target_index = i//31
        old_target_a_number = i%31
        new_target[i][0] = target[old_target_index][old_target_a_number]+1e-30
    target_lb = new_target.min()
    target_ub = new_target.max()
    return new_target,target_lb,target_ub

def delete_zero_index(input,target):
    return input,target

def data_process(input,target):

    input,target = delete_zero_index(input,target)
    input,range_lb,range_ub,range_rate_lb,range_rate_ub,v_lb,v_ub = input_process(input)
    target,target_lb,target_ub = target_process(target)
    return input,target,range_lb,range_ub,range_rate_lb,range_rate_ub,v_lb,v_ub,target_lb,target_ub

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
target = sio.loadmat('Q_table_little.mat')['Q_table_little']
input = sio.loadmat('dangerous_state_table.mat')['dangerous_state_table']
input,target,range_lb,range_ub,range_rate_lb,range_rate_ub,v_lb,v_ub,target_lb,target_ub = data_process(input,target)

#input,target = normalize(input,target,range_lb,range_ub,range_rate_lb,range_rate_ub,v_lb,v_ub,target_lb,target_ub)

print("Input Array Size:",input.shape[0])
#np.save('input.npy',input)
#np.save('target.npy',target)
input = np.load('input.npy')
target = np.load('target.npy')
for i in range(target.shape[0]):
    target[i] = np.sin(input[i,0]+input[i,1]+input[i,2]+input[i,3])
input,target = normalize(input,target,range_lb,range_ub,range_rate_lb,range_rate_ub,v_lb,v_ub,target_lb,target_ub)
test_index = random.sample(range(input.shape[0]), 500) 
test_input = input[test_index]
test_target = target[test_index]
train_index = list(set(range(input.shape[0]))-set(test_index))
train_input = input[train_index]
train_target = target[train_index]
print("Train data size:",train_input.shape[0])
print("Test data size",test_input.shape[0])
print(" ")
print("-----     Setting and netload     -----")
print(" ")
optimizer = torch.optim.Adam(net.parameters(),lr = 0.1)
loss_func = torch.nn.L1Loss()
BATCH_SIZE = 16
print("Using Adam Optim and L1Loss")
torch_dataset = Data.TensorDataset(torch.from_numpy(train_input).cuda(),torch.from_numpy(train_target).cuda())
torch_dataset = torch_dataset
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

last_epoch = -1
if os.path.isfile('./models/model.pth'):
    print("=> loading checkpoint '{}'".format('./models/model.pth'))
    checkpoint = torch.load('./models/model.pth')
    #start_episode = checkpoint['epoch']
    #net.load_state_dict(checkpoint['net'])
    last_epoch = checkpoint['epoch']
    print("=> loaded checkpoint '{}' (epoch {})".format('./models/model.pth', checkpoint['epoch']))

print(" ")
print("-----     Begin Training    -----")
print(" ")
  
for epoch in range(last_epoch+1,10000000000000):
    
    for step,(data_step,target_step) in enumerate(loader):
        #print(step)
        #print(data_step)
        prediction = net(data_step)
        # print(prediction)
        loss = loss_func(prediction,target_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch * len(loader) + step)%1e4 == 9999:
            net_state = {'net':net.state_dict(), 'epoch':epoch}
            torch.save(net_state,'./models/model.pth')
            print("=> saving checkpoint at epoch",epoch)
        if (epoch * len(loader) + step)%1e3 == 999:
            print(net(data_step[0,:]))
            print(prediction[0])
            print("epoch:",epoch,"batch:",step,"loss:",loss)
        if (epoch * len(loader) + step)%10 == 0:
            niter = epoch * len(loader) + step
            writer.add_scalar("Train/Loss",loss,niter)
        if (epoch * len(loader) + step)%500 == 0:
            niter = epoch * len(loader) + step
            myloss = apply_test(torch.from_numpy(test_input).cuda(),torch.from_numpy(test_target).cuda())
            #print("episode:",step,"test_loss:",myloss)
            writer.add_scalar('Test/Accu', myloss, niter)
        
        

    
    
