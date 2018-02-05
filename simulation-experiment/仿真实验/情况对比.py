# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2017/12/14 20:42
'''
import numpy as np
import matplotlib.pyplot as plt

theta1_1=np.load('data\Theta1_1.npy')
theta2_1=np.load('data\Theta2_1.npy')
theta1_2=np.load('data\Theta1_2.npy')
theta2_2=np.load('data\Theta2_2.npy')
step=np.load('data\step.npy')
plt.figure(1)
plt.plot(step,theta1_1,label='theta1_1')
plt.plot(step,theta1_2,label='theta2_1')
plt.plot(step,theta2_1,label='theta1_2')
plt.plot(step,theta2_2,label='theta2_2')
plt.legend()
plt.show()