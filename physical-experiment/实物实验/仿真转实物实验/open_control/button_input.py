# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/3/28 10:50
'''

#按钮控制程序开始
import PyDAQmx
import numpy as np

from PyDAQmx import *


class BUTTON(object):
    def __init__(self):
        self.task0 = Task()
        self.read3 = int32()
        self.data3 = np.zeros((3,), dtype=np.uint8)
        self.task0.CreateDIChan("/Dev2/port1/line0:3","",PyDAQmx.DAQmx_Val_ChanForAllLines)
        self.task0.StartTask()

    def state(self):
        while True:
            self.task0.ReadDigitalU8(1, 10.0, DAQmx_Val_GroupByChannel, self.data3, 3, byref(self.read3), None)
            state=self.data3[0]#state等于15时为松开状态，14时为按下状态
            if state==15:
                break
        self.task0.StopTask()