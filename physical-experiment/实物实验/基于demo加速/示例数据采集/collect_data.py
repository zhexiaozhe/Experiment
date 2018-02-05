# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/11 9:54
'''
import gym
from pygame import *
import pygame

#########环境############
env=gym.make('Acrobot-v1')
env.reset()
##########手柄###########
pygame.init()
j = pygame.joystick.Joystick(0)
j.init()
action=0
##########数据##########
data=[]
for step in range(2000):
    env.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:
                action = 10*event.value
    obs,r,done,inf=env.step([action])

    if done:
        break