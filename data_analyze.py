import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

def normalize(input,target):
    input[:,0] = (input[:,0]-np.mean(input[:,0]))/np.std(input[:,0])
    input[:,1] = (input[:,1]-np.mean(input[:,1]))/np.std(input[:,1])
    input[:,2] = (input[:,2]-np.mean(input[:,2]))/np.std(input[:,2])
    input[:,3] = (input[:,3]-np.mean(input[:,3]))/np.std(input[:,3])
    target = (target-np.mean(target))/np.std(target)
    return input,target

input = np.load('input.npy')
target = np.load('target.npy')
data_size = input.shape[0]
#input,target = normalize(input,target)

ax = plt.figure().add_subplot(111, projection = '3d') 
#基于ax变量绘制三维图 
#xs表示x方向的变量 
#ys表示y方向的变量 
#zs表示z方向的变量，这三个方向上的变量都可以用list的形式表示 
#m表示点的形式，o是圆形的点，^是三角形（marker) 
#c表示颜色（color for short） 
ax.scatter(input[:,0], input[:,1], input[:,2], c = 'r', marker = '^') #点为红色三角形 
 
#设置坐标轴 
ax.set_xlabel('range Label') 
ax.set_ylabel('range rate Label') 
ax.set_zlabel('v Label') 
 
#显示图像 
plt.show() 