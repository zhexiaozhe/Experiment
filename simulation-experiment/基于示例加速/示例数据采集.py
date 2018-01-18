# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/17 10:03
'''
import pygame
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
import filter_env

ENV_NAME='Acrobot-v1'
env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
# pygame.init()
# j = pygame.joystick.Joystick(0)
# j.init()
# action = 0
i=0
demo_buffer=[]
for i in range(2000):
    state=env.reset()
    E=[]
    for step in range(1000):
        # env.render()
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         exit()
        #     elif event.type == pygame.JOYAXISMOTION:
        #         if event.axis == 0:
        #             action = 10*event.value
        action=np.resize(6/10*np.sin(0.02*np.pi*step),[1])
        next_state,reward,done,inf=env.step(action)
        experience = (state, action, reward, next_state, done)
        demo_buffer.append(experience)
        state = next_state
        if done:
            print('episode:%s step: %s'%(i,step))
            break

np.save('data/demo_buffer.npy',demo_buffer)
