import numpy as np 
import matplotlib.pyplot as plt

input = np.load('input.npy')
target_data = np.load('target.npy')
np.savetxt("input.csv", input, delimiter=',')
np.savetxt("target.csv", target_data, delimiter=',')
whole_data = np.hstack((input,target_data))
whole_data = whole_data[np.argsort(whole_data[:,4])]
whole_data = whole_data[-13000:,]
input = whole_data[:,0:4]
target_data = whole_data[:,4]
range_data = input[:,0]
range_rate_data = input[:,1]
v_data = input[:,2]
a_data = input[:,3]
np.save("input.npy",input)
np.save("target.npy",target_data)
plt.hist(target_data[-13000:],200)
plt.show()
print(target_data[-13000])
print("over")