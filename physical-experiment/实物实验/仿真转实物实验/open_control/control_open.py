# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/3/28 9:12
'''
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt

from PyDAQmx import *
from button_input import *
from sensor import *
from distance import *

if __name__=='__main__':

    pygame.init()
    j = pygame.joystick.Joystick(0)
    j.init()

    task0 = Task()
    task0.CreateDOChan("/Dev2/port0/line2:7", "", PyDAQmx.DAQmx_Val_ChanForAllLines)
    task0.StartTask()
    data0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    paw_state = 0                                                                                                        #paw_state=0时为张开，=1时为闭合
    print('拉到初始位置，并按下A键闭合手抓')

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 1:
                    data0 = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8)
                    paw_state = 1
        task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)
        if paw_state == 1:
            break

    done = 0
    print('先按下按钮，然后按手柄B键开始程序...')

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 2:
                    done=1
        if done==1:
            break

    button=BUTTON()
    button.state()
    sen = CONTROL()
    sen.start()
    d=DISTANCE()
    Action=np.load(r'data\torque1.npy')
    data0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    start=time.clock()
    Dis=[]
    T_get=[]
    T_send=[]
    Theta1=[]
    Theta2=[]
    step=0
    for action in Action:
        # 系统状态采集
        angle1, angle_velocity1 = sen.read_ahrs()
        angle2, angle_velocity2, torque = sen.read_daq()
        state=[angle1,angle2]
        Theta1.append(state[0])
        Theta2.append(state[1])
        D=d.dis(state)
        T_send.append(10*action)
        if step > 0:
            T_get.append(torque)
        step += 1
        Dis.append(D)
        print('步数：',step,'距离：',D)
        # 发送控制力矩
        sen.write_daq(10/2.73*action)
        # 控制手抓开合
        if D<=0.1:
            data0 = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8)
            task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)
            break
        task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)
    print('时间：',time.clock()-start)
    sen.stop()

    plt.figure('姿态')
    plt.plot(Theta1,label='Theta1')
    plt.plot(Theta2,label='Theta2')
    plt.legend()

    plt.figure('距离')
    plt.plot(Dis, label="distance")
    plt.legend()

    plt.figure('动作')
    plt.plot(T_send,label="Torque_send")
    plt.plot(T_get, label="Torque_get")
    plt.legend()
    plt.show()