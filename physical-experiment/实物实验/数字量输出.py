# encoding: utf-8
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 采集卡数字量输出.py
@time: 2017/9/26 16:20
'''
from PyDAQmx import Task
import PyDAQmx
import numpy as np


data = np.array([0,0,0,0,0,0,0,0], dtype=np.uint8)

task = Task()
task.CreateDOChan("/Dev2/port0/line0:7","",PyDAQmx.DAQmx_Val_ChanForAllLines)
task.WriteDigitalLines(1,1,10.0,PyDAQmx.DAQmx_Val_GroupByChannel,data,None,None)
task.StopTask()