# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: modle_test.py
@time: 2017/11/2 11:14
'''
import math
import matplotlib
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

env=gym.make('Acrobot-v1')
# env=gym.make('Pendulum-v0')
# env=gym.make('Pendulum-v0')
fontsize=10
env.reset()
env=env.unwrapped
theta1=[]
theta2=[]
dtheta1=[]
dtheta2=[]
Energy=[]
times=[]
for step in range(500):
    # image=env.render(mode='rgb_array')
    # print(np.shape(image))
    env.render()
    time.sleep(120)
    # time.sleep(0.1)
    action = [0]
    # if step<50:
    #     action=[1]
    # else:
    #     action=[0]
    ob,r,done,inf=env.step(action)
    # if done:
    #     break
    # matplotlib.image.imsave(r'figure\name.png', image)
    theta1.append(inf[0][0])
    theta2.append(inf[0][1])
    Energy.append(inf[5])
    times.append(0.02*step)
dataFile = u'F:/matlab_files/test2.mat'
sio.savemat(dataFile, {'Theta1':theta1,'Theta2':theta2,'Time':times,'Energy':Energy})

plt.figure('响应曲线')
plt.plot(times,theta1,'r--',label=r'$\theta_1$')
plt.plot(times,theta2,'b-',label=r'$\theta_2$')
plt.legend()
plt.xlabel('Time/s',fontsize=fontsize)
plt.ylabel('Angle/rad',fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.figure('能量')
plt.plot(times,Energy,'b-')
plt.xlabel('Time/s',fontsize=fontsize)
plt.ylabel('Energy/J',fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()