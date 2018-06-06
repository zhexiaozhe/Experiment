# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/11 18:41
'''
import PyDAQmx
from PyDAQmx import *
from pygame import *
import pygame
import numpy as np

task0 = Task()
task0.CreateDOChan("/Dev2/port0/line0:7","",PyDAQmx.DAQmx_Val_ChanForAllLines)
task0.StartTask()
data0=np.array([0,0,0,0,0,0,0,0],dtype=np.uint8)

#############手柄##############
pygame.init()
j = pygame.joystick.Joystick(0)
j.init()
action=0
done=0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button==3:
                data0=np.array([0,0,1,0,0,0,0,0], dtype=np.uint8)
            if event.button==8:  #back按键是退出键
                done=1
        elif event.type==pygame.JOYBUTTONUP:
            if event.button == 3:
                data0=np.array([0,0,0,0,0,0,0,0],dtype=np.uint8)
    task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)

    if done:
        task0.StopTask()
        break