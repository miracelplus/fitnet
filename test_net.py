import numpy as np
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.data as Data
from tensorboardX import SummaryWriter
writer = SummaryWriter('mylog')

def normalize(input,target):
    input[:,0] = (input[:,0]-np.mean(input[:,0]))/np.std(input[:,0])
    input[:,1] = (input[:,1]-np.mean(input[:,1]))/np.std(input[:,1])
    target = (target-np.mean(target))/np.std(target)
    return input,target

class Net(nn.Module):
    
    def __init__(self,):
        N_STATES = 2
        N_VALUE = 1
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
        #print(x)
        x = self.out(x)
        #print(x)
        return x

net = Net().cuda()
test_num = int(1e5)
input = np.linspace(-1000,1000,1e5).reshape((100000,1))
input2  = np.linspace(2000,3000,1e5).reshape((100000,1))
input = np.hstack((input,input2))
target = np.sin(input)
input,target = normalize(input,target)
optimizer = torch.optim.Adam(net.parameters(),lr = 0.1)
loss_func = torch.nn.L1Loss()
torch_dataset = Data.TensorDataset(torch.from_numpy(input).cuda(),torch.from_numpy(target).cuda())
torch_dataset = torch_dataset
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size= 16,
    shuffle=True
)

for epoch in range(100):
    for step,(data_step,target_step) in enumerate(loader):
        prediction = net(data_step)
        loss = loss_func(prediction,target_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch * len(loader) + step)%10 == 0:
            niter = epoch * len(loader) + step
            writer.add_scalar("Train/Loss",loss,niter)
        
