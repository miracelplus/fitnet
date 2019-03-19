import numpy as np
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import random
writer = SummaryWriter('mylog')

# torch.manual_seed(1)    # reproducible

x1 = (2*torch.rand(50000,1)-1).cuda()  # x data (tensor), shape=(100, 1)
x2 = (2*torch.rand(50000,1)-1).cuda()
x3 = (2*torch.rand(50000,1)-1).cuda()  # x data (tensor), shape=(100, 1)
x4 = (2*torch.rand(50000,1)-1).cuda()
xm = torch.cat((x3,x4),1)
x = torch.cat((x1,x2),1)
x = torch.cat((x,xm),1)
x_cankao = x1+x2+x3+x4
y = torch.sin(3*x_cankao)                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


def normalize(input,target):
    input[:,0] = (input[:,0]-np.mean(input[:,0]))/np.std(input[:,0])
    input[:,1] = (input[:,1]-np.mean(input[:,1]))/np.std(input[:,1])
    input[:,2] = (input[:,2]-np.mean(input[:,2]))/np.std(input[:,2])
    input[:,3] = (input[:,3]-np.mean(input[:,3]))/np.std(input[:,3])
    target = (target-np.mean(target))/np.std(target)
    return input,target

def normalize_torch(input,target):
    input[:,0] = (input[:,0]-torch.mean(input[:,0]))/torch.std(input[:,0])
    input[:,1] = (input[:,1]-torch.mean(input[:,1]))/torch.std(input[:,1])
    input[:,2] = (input[:,2]-torch.mean(input[:,2]))/torch.std(input[:,2])
    input[:,3] = (input[:,3]-torch.mean(input[:,3]))/torch.std(input[:,3])
    target = (target-torch.mean(target))/torch.std(target)
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
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        #print(x.grad)
        '''
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        '''
        x = self.predict(x)             # linear output
        print(x)
        return x

net = Net(n_feature=4, n_hidden=50, n_output=1).cuda()     # define the network
print(net)  # net architecture
x_lizi = np.linspace(-4,4,500)
y_lizi = np.sin(3*x_lizi)

lr = 0.1
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss_func = torch.nn.L1Loss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

x = torch.from_numpy(np.load('input.npy')).cuda()
y = torch.from_numpy(np.load('target.npy')).cuda()
index_num = 50
data_index = random.sample(range(x.shape[0]), index_num)
#data_index = [100,111,123,124,126,178,270,220,440,450]
x = x[data_index]
y = y[data_index].reshape(index_num,1)
'''
data_index = random.sample(range(x.shape[0]), 500)
x = x[data_index]
y = y[data_index]
'''
x,y = normalize_torch(x,y)
#x = torch.from_numpy(x).cuda()
#y = torch.from_numpy(y).cuda()
x_cankao = x.sum(1)
whole_test_num = int(2e9)
for t in range(int(whole_test_num)):
    #if t%30 == 0:
    lr = 0.1*(whole_test_num-t)/whole_test_num
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    print("epoch:",t)
    prediction = net(x)     # input x and predict based on x
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    print("loss:",loss.data.cpu().numpy(),"| lr:",lr)
    #print('x:',x,'y:',y,'prediction:',prediction)
    #if t % 10 == 0:
    #    writer.add_scalar('Test/Accu', loss, t)
    if t % 10 == 0:
        # plot and show learning process
        plt.cla()
        plt.ylim(-1.5,1.5)
        plt.plot(x_lizi,y_lizi)
        #plt.scatter(x_cankao.data.cpu().numpy(), y.data.cpu().numpy())
        plt.scatter(x_cankao.data.cpu().numpy(), prediction.data.cpu().numpy(),c='r',marker='.')
        plt.text(0, 1.3, 'Loss=%.4f' % loss.data.cpu().numpy(), fontdict={'size': 14, 'color':  'red'})
        plt.pause(0.1)
        writer.add_scalar("Train/Loss",loss,t)
        params = list(net.named_parameters())
        print('-------------------------------------------------')
        (name, param) = params[17]
        print(name)
        print(param.grad)
        print('-------------------------------------------------')
        (name, param) = params[18]
        print(name)
        print(param.grad)
plt.ioff()
plt.show()
