import numpy as np 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDerivationx(y):
    return y*(1-y)

if __name__ == "__main__":
    bias = [0.35,0.60]
    weight = [0.15,0.2,0.25,0.3,0.4,0.45,0.5,0.55]
    output_layer_weights = [0.4,0.45,0.5,0.55]
    i1 = 0.05
    i2 = 0.10
    target1 = 0.01
    target2 = 0.99
    alpha = 0.5
    numIter = 10000
    for i in range(numIter):
        #正向传播
        neth1 = i1*weight[1-1] + i2*weight[2-1] + bias[0]
        neth2 = i1*weight[3-1] + i2*weight[4-1] + bias[0]
        outh1 = sigmoid(neth1)
        outh2 = sigmoid(neth2)
        neto1 = outh1*weight[5-1] + outh2*weight[6-1] + bias[1]
        neto2 = outh2*weight[7-1] + outh2*weight[8-1] + bias[1]
        outo1 = sigmoid(neto1)
        outo2 = sigmoid(neto2)