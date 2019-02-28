import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CF_ENV import *
from functions import *


# Hyper Parameters
BATCH_SIZE = 32
LR = 1e-5                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 1
N_STATES = 3
env2 = CF_ENV()
N_ACTIONS = len(env2.action_space)



class Net(nn.Module):
    '''
    神经网络类
    '''
    def __init__(self, ):
        '''
        网络结构：N_STATES -> 50 -> relu -> N_ACTIONS -> sigmoid -> N_ACTIONS 全连接网络
        '''
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0)   # initialization
        self.fc2 = nn.Linear(50, 50)
        self.fc2.weight.data.normal_(0, 0)
        self.fc3 = nn.Linear(50, 50)
        self.fc3.weight.data.normal_(0, 0)
        self.fc4 = nn.Linear(50, 50)
        self.fc4.weight.data.normal_(0, 0)
        self.fc5 = nn.Linear(50, 50)
        self.fc5.weight.data.normal_(0, 0)
        self.fc6 = nn.Linear(500, 500)
        self.fc6.weight.data.normal_(0, 0)
        self.fc7 = nn.Linear(500, 500)
        self.fc7.weight.data.normal_(0, 0)
        self.fc8 = nn.Linear(500, 500)
        self.fc8.weight.data.normal_(0, 0)
        self.fc9 = nn.Linear(500, 500)
        self.fc9.weight.data.normal_(0, 0)
        self.fc10 = nn.Linear(500, 500)
        self.fc10.weight.data.normal_(0, 0)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0)   # initialization

    def forward(self, x):
        '''
        网络的前向传播函数，含有一个relu激活函数以及一个sigmoid激活函数
        '''
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.out(x)
        actions_value = F.relu(x)
        return actions_value

 
class DQN(object):
    '''
    DQN类，也就是agent类
    '''
    def __init__(self):
        '''
        建立评估网络以及现实网络
        '''
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        #选择动作 训练网络使用 随机采样
        #x = torch.unsqueeze(torch.FloatTensor(x), 0)
        #actions_value = self.eval_net.forward(x)
        action = np.random.randint(0, N_ACTIONS)
        #action = action
        #print(type(action))
        return action

    def choose_action_test(self, x):
        #选择动作 实际测试使用  greedy采样
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0]
        return action
        
    def get_state_collision_rate(self,x):
        #sum up the output of the state and get collision rate
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        actions_value = self.eval_net.forward(x)
        #print(type(torch.sum(actions_value).item()))
        return torch.sum(actions_value).item()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        #如果记忆区已经满了就覆盖旧有的记忆
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 更新目标网络target_net(更新的比较慢的那个) 同时记录学习的次数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        #选取本次学习的批次
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        # q_eval是一个32×1的tensor，从32×31的tensor提取而来
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next没有提取，大小为32*31
        q_myeval = torch.unsqueeze(torch.tensor(np.sum(q_next.numpy(),axis=1)),1)
        q_target = torch.randn_like(q_eval)
        mmm = b_a[0].numpy()[0]
        mmm = b_s[0].numpy()[2]
        #print(get_NDD_possi(b_a[0].numpy()[0],b_s[0,2].numpy()[0]))
        for i in range(BATCH_SIZE):
            
            if b_r[i].numpy()[0] == 1:
                q_target[i] = get_NDD_possi(b_a[i].numpy()[0],b_s[i].numpy()[2])
            elif b_r[i].numpy()[0] == -1:
                q_target[i] = 0
            else:
                q_target[i] = q_myeval[i]*get_NDD_possi(b_a[i,0].item(),b_s[i,2].item())
        #q_target = b_r + q_myeval.view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        #print(q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    '''
    def learn_one_step(self,s, a, r, s_):
        q_eval = self.eval_net(s).gather(1, a)  # shape (batch, 1)
        q_next = self.target_net(s_).detach()     # detach from graph, don't backpropagate
        q_myeval = torch.unsqueeze(torch.tensor(np.sum(q_next.numpy(),axis=1)),1)
        q_target = r + q_myeval.view(BATCH_SIZE, 1)   # shape (batch, 1)
        
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    '''