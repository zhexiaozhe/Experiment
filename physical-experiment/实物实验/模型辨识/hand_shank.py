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
        try:
            pygame.init()
            j = pygame.joystick.Joystick(0)
            j.init()
            self.action = 0
            print('手柄连接成功')
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