# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/20 16:31
'''

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt


MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'



if __name__ =='__main__':
    #生成简单数据
    np.random.seed(1)
    X=np.linspace(1,10,100)
    Y=0.5*np.random.random(size=100)
    plt.figure('训练数据')
    plt.scatter(X,Y)
    plt.show()


