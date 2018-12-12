# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/11/18 21:44
'''
import matplotlib.pyplot as plt
import gym
from numpy import sin,pi
from save_data import SAVE_DATA
# from plotting import PLOT

env=gym.make('Acrobot-v1')
env.reset()
save_data=SAVE_DATA()
# plt=PLOT()
theta1=[]
theta2=[]
dtheta1=[]
dtheta2=[]
data=save_data.load()
Theta1=data[0]
Theta2=data[1]
for step in range(1000):
    # env.render()
    if step < 200:
        value = 0
    else:
        value = 7 *sin(0.01 * pi * (step - 200))
    ob, r, done, inf = env.step([value])
    theta1.append(inf[0][0])
    theta2.append(inf[0][1])

plt.figure('响应曲线')
plt.plot(theta1,'r-',label=r'$\theta_1$')
plt.plot(theta2,'b-',label=r'$\theta_2$')
plt.plot(Theta1,'y--',label=r'$\theta_1$_py')
plt.plot(Theta2,'g-',label=r'$\theta_2$_py')
plt.legend()
plt.grid()
plt.show()
# plt.plot(save_data.load())

