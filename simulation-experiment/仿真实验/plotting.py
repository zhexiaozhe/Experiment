# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/6/28 9:00
'''

import filter_env
import time
import gc
import matplotlib.pyplot as plt
import numpy as np
import gym
from ddpg import *
plt.rcParams['font.sans-serif']=['SimHei']#增加中文功能
plt.rcParams['axes.unicode_minus']=False
gc.enable()

ENV_NAME = 'Acrobot-v1'
test_name='论文实验1'

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    state = env.reset()
    total_reward = 0
    theta1=[]
    theta2=[]
    theta1d=[]
    theta2d=[]
    st=[]
    distance=[]
    action=[]
    smooth_action=0
    for step in range(env.spec.timestep_limit):
        # env.render()

        action = agent.action(state)
        if step==0:
            smooth_action=action
        else:
            smooth_action=0.5*smooth_action+0.5*action
        state, reward, done, inf = env.step(smooth_action)
        total_reward += reward
        if inf[0][0]>=3:
            inf[0][0]=inf[0][0]-2*np.pi
        theta1.append(inf[0][0])
        theta2.append(inf[0][1])
        theta1d.append(inf[1])
        theta2d.append(inf[2])
        distance.append(inf[5])
        st.append(0.01*step)
    print(total_reward)
    plt.figure(1)
    plt.plot(st,theta1, 'r--', label='theta1')
    plt.plot(st,theta2, 'b-', label='theta2')
    plt.plot(st,theta1d, 'g-.', label='theta1d')
    plt.plot(st,theta2d, 'c:', label='theta2d')
    plt.xlabel('time/s')
    plt.ylabel('angle/rad')
    plt.grid()
    plt.legend()
    plt.figure(2)
    plt.plot(distance)
    plt.grid()
    plt.show()
if __name__ == '__main__':
    main()
