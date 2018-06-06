# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/5/14 9:41
'''

import gym
import pygame
import pickle
import numpy as np

from collections import deque
from sys import exit

class HAND_SHANK(object):
    def __init__(self):
        try:
            pygame.init()
            j = pygame.joystick.Joystick(0)
            j.init()
            self.action = 0
            print('手柄连接成功')
        except pygame.error :
            print('手柄连接失败')
            exit()

    def control(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.JOYAXISMOTION:
                if event.axis == 0:
                    self.action = event.value
        return [10*self.action]

if __name__ =='__main__':
    hand=HAND_SHANK()
    env = gym.make('Acrobot-v1')
    buffer=deque()
    for episode in range(50):
        state=env.reset()
        normal=[1,1,1,1,4*np.pi,9*np.pi]
        for step in range(env.spec.timestep_limit):
            env.render()
            action=hand.control()
            next_state, r, done, inf = env.step(action)
            buffer.append((state/normal,action,r,next_state/normal,done))
            if done==1:
                print(step)
                break
            state = next_state
    pickle.dump(buffer, open('object.pickle', 'wb'))
    # np.save('buffer.npy',buffer)