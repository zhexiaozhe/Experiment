# encoding: utf-8
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 编码器数据采集.py
@time: 2017/9/29 8:50
'''

from __future__ import print_function
from PyDAQmx import *
import numpy
import time

task=Task()
read = int32()
data = numpy.zeros((1,), dtype=numpy.float64)
task.CreateCICountEdgesChan("Dev2/ctr0","",DAQmx_Val_Falling,0,DAQmx_Val_CountUp)
# task.CfgSampClkTiming("",1.0,DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,1)#暂时不需要
task.StartTask()
while 1:
    task.ReadCounterF64(1,1.0,data,1,byref(read),None)
    angle=data[0]*(360/102400)#现在显示的是杆2真实的角度
    print("杆2与杆1的夹角:%f 度"%angle)
