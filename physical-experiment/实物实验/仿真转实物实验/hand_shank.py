# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2017/12/24 9:34
'''

import pygame
from sys import exit

class HAND_SHANK(object):
    def __init__(self):
        self.com=0
        try:
            pygame.init()
            j = pygame.joystick.Joystick(0)
            j.init()
            self.action = 0
            print('手柄连接成功')
            print('先按下按钮，然后按下手柄A键')
        except pygame.error :
            print('手柄连接失败')
            exit()
    @property
    def control(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.JOYAXISMOTION:
                if event.axis == 0:
                    self.action = 10 / 2.73 * event.value
        return self.action

    def command(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type==pygame.JOYBUTTONDOWN:
                if event.button==1:
                    self.com = 1
                    break
                elif event.button==2:
                    self.com=2
                    break
        return self.com