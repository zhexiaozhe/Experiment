# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 角度二位置采集.py
@time: 2017/10/8 19:43
'''
import numpy as np
from PyDAQmx import *
import time
from numpy import pi,sin,cos

sgn=lambda x:1 if x>0 else -1 if x<0 else 0

task=Task()
read = int32()
data = np.zeros((1,), dtype=numpy.float64)
#角速度采集

task.CreateAIVoltageChan("Dev2/ai0","",DAQmx_Val_Cfg_Default,-10.0,10.0,DAQmx_Val_Volts,None)
#角度采集
task1=Task()
read1 = int32()
data1 = numpy.zeros((1,), dtype=numpy.float64)
task1.CreateCICountEdgesChan("Dev2/ctr0","",DAQmx_Val_Falling,0,DAQmx_Val_CountUp)
task.StartTask()
task1.StartTask()
angle=0
pre_data1=np.zeros((1,), dtype=numpy.float64)
start_time=time.clock()
for i in range(10000):
    task.ReadAnalogF64(1, 1.0, DAQmx_Val_GroupByChannel, data, 1, byref(read), None)
    task1.ReadCounterF64(1, 1.0, data1, 1, byref(read1), None)
    dir=sgn(data)
    dif_data1=data1[0]-pre_data1
    dif_angle=dir*dif_data1*(360/102400)*pi/180
    angle=angle+dif_angle
    pre_data1=data1[0]
    print(angle)
    time.sleep(0.02)
print(time.clock()-start_time)