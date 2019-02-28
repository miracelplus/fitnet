from functions import *


def sample_range():
    pass

def sample_range_rate():
    pass

def sample_v():
    pass

def update_state(state,a,dt,limit):
    range_tmp = state[0]
    range_rate_tmp = state[1]
    v1 = state[2]
    v2 = v1 - range_rate_tmp
    x1 = range_tmp
    x2 = 0
    a2 = IDM(x1,v1,x2,v2)
    a1 = a
    state1 = update_xv(x1,v1,a1,dt,limit)
    state2 = update_xv(x2,v3,a2,dt,limit)
    state[0] = state1[0]-state2[0]
    state[1] = state1[1]-state2[1]
    state[2] = state1[1]
    return state

whole_test_num = 1e6
test_num = 300
dt = 0.1
final_possi_list = []
likelyhood = []
a_label = -2:0.2:4

possi_list_Q = []
like_list = []

for episode in range(whole_test_num):
    print("episode:"+i)
    test_range = sample_range()
    test_range_rate = sample_range_rate()
    test_v = sample_v()
    state = [test_range,test_range_rate,test_v]
    NDD_ini_possi = get_NDD_state_possi(state)
    Q_ini_possi = get_Q_state_possi(state)
    like = NDD_ini_possi/Q_ini_possi

    for step in range(test_num):
        flag = 1
        NDD_a_possi_list = get_NDD_a_possi_list(state)
        Q_a_possi_list = get_Q_a_possi_list(state)
        test_a_num = sample_a_in_list(Q_a_possi_list)
        test_a = a_label[test_a_num]
        Q_possi = Q_a_possi_list[test_a_num]
        NDD_possi = NDD_a_possi_list[test_a_num]
        state,reward = update_state(state,test_a)
        like = like*NDD_possi/Q_possi

        if reward = 1:
            flag = 0
        
    if flag == 0:
        possi_list_Q[episode] = 1
        like_list[episode] = like
    else:
        possi_list_Q[episode] = 0
        like_list[episode] = like


        


