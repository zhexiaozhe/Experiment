# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/4/1 10:03
'''
import numpy as np
import PyDAQmx

from PyDAQmx import *

class PAW(object):
    def __init__(self):
        self.task0 = Task()
        self.task0.CreateDOChan("/Dev2/port0/line2:7", "", PyDAQmx.DAQmx_Val_ChanForAllLines)
        self.data0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    def start(self):
        self.task0.StartTask()

    def open(self):
        self.data0 = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, self.data0, None, None)

    def close(self):
        self.data0=np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, self.data0, None, None)