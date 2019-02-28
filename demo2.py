import numpy as np 
import matplotlib.pyplot as plt

input = np.load('input.npy')
target_data = np.load('target.npy')

range_data = input[:,0]
range_rate_data = input[:,1]
v_data = input[:,2]
a_data = input[:,3]

plt.hist(target_data)
plt.show()
print("over")