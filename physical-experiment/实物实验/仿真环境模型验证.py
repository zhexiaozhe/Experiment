# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 仿真环境模型验证.py
@time: 2017/10/9 14:58
'''

import gym
import numpy as np
from numpy import pi,sin,cos
import matplotlib.pyplot as plt
import scipy.io as sio

env=gym.make('Acrobot-v1')
env.reset()
A=[]
Angle1=[]
Angle2=[]
Angle_velocity1=[]
Angle_velocity2=[]
E=[]

#加载外部的力矩数据
matfn=u'F:/matlab_file/xiaogu.mat'
matdata=sio.loadmat(matfn)
t1=matdata['torque01']

A1=[]
for step in range(501):
    env.render()
    # a=t1[step]
    if step<101:
        a=[0]
    else:
        a=[6*sin(0.02*pi*(step-101))]
    # a=[3]
    obs,r,done,inf=env.step(a)
    Angle1.append(inf[2][0])
    Angle2.append(inf[2][1])
    Angle_velocity1.append(inf[2][2])
    Angle_velocity2.append(inf[2][3])
    A.append(a)
    # A1.append(a1)
    # E.append(inf[6])
# plt.plot(E)

#仿真数据采集
np.save('data\S_Theta1.npy',np.array(Angle1))
np.save('data\S_Theta2.npy',np.array(Angle2))
np.save('data\S_torque.npy',np.array(A))

plt.figure(1)
plt.title('Action Figure')
plt.xlabel('Step/0.02s')
plt.ylabel('Torque/N.m')
plt.plot(A,'r--',label='$torque$')
# plt.plot(A1,'b--',label='$torque1$')
plt.grid()
plt.legend()
plt.figure(2)

plt.title('Angle Figure')
plt.xlabel('Step/0.02s')
plt.ylabel('Angle/rad')
plt.plot(Angle1,'r--',label='$theta1$')
plt.plot(Angle2,'b--',label='$theta2$')
plt.legend()
plt.grid()

plt.figure(3)
plt.title('Angle_valocity Figure')
plt.xlabel('Step/0.02s')
plt.ylabel('Angle_valocity/rad/s')
plt.plot(Angle_velocity1,'r-',label='$dtheta1$')
plt.plot(Angle_velocity2,'g-',label='$dtheta2$')
plt.legend()
plt.grid()

plt.show()
