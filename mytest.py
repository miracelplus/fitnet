import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
writer = SummaryWriter('mylog')

# torch.manual_seed(1)    # reproducible

x1 = (2*torch.rand(500,1)-1).cuda()  # x data (tensor), shape=(100, 1)
x2 = (2*torch.rand(500,1)-1).cuda()
x3 = (2*torch.rand(500,1)-1).cuda()  # x data (tensor), shape=(100, 1)
x4 = (2*torch.rand(500,1)-1).cuda()
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
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = F.relu(self.hidden6(x))
        x = F.relu(self.hidden7(x))
        x = F.relu(self.hidden8(x))
        x = F.relu(self.hidden9(x))
        x = self.predict(x)             # linear output
        #print(x)
        return x

net = Net(n_feature=4, n_hidden=500, n_output=1).cuda()     # define the network
print(net)  # net architecture
x_lizi = np.linspace(-4,4,500)
y_lizi = np.sin(3*x_lizi)

lr = 0.3
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss_func = torch.nn.L1Loss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(int(2e6)):
    if t%200 == 0:
        lr = lr*0.95
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    print("epoch:",t)
    prediction = net(x)     # input x and predict based on x
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    print("loss:",loss.data.cpu().numpy(),"| lr:",lr)
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.ylim(-1.5,1.5)
        plt.plot(x_lizi,y_lizi)
        #plt.scatter(x_cankao.data.cpu().numpy(), y.data.cpu().numpy())
        plt.scatter(x_cankao.data.cpu().numpy(), prediction.data.cpu().numpy(),c='r',marker='.')
        plt.text(0, 1.3, 'Loss=%.4f' % loss.data.cpu().numpy(), fontdict={'size': 14, 'color':  'red'})
        plt.pause(0.1)
        writer.add_scalar("Train/Loss",loss,t)

plt.ioff()
plt.show()