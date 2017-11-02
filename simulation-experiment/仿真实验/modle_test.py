# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: modle_test.py
@time: 2017/11/2 11:14
'''
import gym
import time
import numpy as np

env=gym.make('Acrobot-v1')
env.reset()
for step in range(1000):
    env.render()
    ob,r,done,inf=env.step([0])
    print(inf[2])