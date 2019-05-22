
import numpy as np
import matplotlib.pyplot as plt

cost=np.load('/Users/fengqi/Pycharm_py36/QF/lrud_cost.npy')
avg=cost*0
ind=25000
for ii in range (ind):
    avg[ii]=np.average(cost[ii:ii+10])
plt.close('all')
plt.plot(avg[:ind-100])
plt.ylim([0,0.1])
