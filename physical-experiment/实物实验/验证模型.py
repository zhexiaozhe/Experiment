# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 验证模型.py
@time: 2017/10/10 19:18
'''

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from numpy import pi,cos,sin

#读取MATLAB数据,目的是对登科的数据进行验证
matfn=u'F:/matlab_file/q.mat'
matdata=sio.loadmat(matfn)
theta1=matdata['Theta1']
theta2=matdata['Theta2']
theta1=theta1.reshape([501,1])
theta2=theta2.reshape([501,1])
#系统采集
e_theta1=np.load("data\Theta1.npy")
e_theta2=np.load("data\Theta2.npy")
s_theta1=np.load("data\S_Theta1.npy")
s_theta2=np.load("data\S_Theta2.npy")

# plt.plot(e_theta1,'m-',label='e_theta1')
# plt.plot(e_theta2,'c-',label='e_theta2')
plt.plot(theta1,'r-',label='theta1')
plt.plot(theta2,'b-',label='theta2')
plt.plot(s_theta1,'g--',label='$simulation-theta1$')
plt.plot(s_theta2,'y--',label='$simulation-theta2$')
plt.grid()
plt.legend()
plt.show()
