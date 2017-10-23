# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 采集卡模拟量数据采集.py
@time: 2017/9/30 10:19
'''

from PyDAQmx import *
from numpy import pi


read = int32()
data = numpy.zeros((2,), dtype=numpy.float64)
#角速度采集
task=Task()
task.CreateAIVoltageChan("Dev2/ai0:1","",DAQmx_Val_Cfg_Default,-10.0,10.0,DAQmx_Val_Volts,None)
# task.CfgSampClkTiming("",10000.0,DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,10)
task.StartTask()
task.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data, 2, byref(read), None)
begin_torque=data[1]*2.73
while 1:
    #角速度采集
    task.ReadAnalogF64(1,10.0,DAQmx_Val_GroupByChannel,data,2,byref(read),None)
    angular_velocity = data[0] * pi / 3
    torque = data[1] * 2.73 - begin_torque
    print("角速度为：%f rad/s"%angular_velocity)
    # print("电机的扭矩为：%f N·m" %torque)
task.StopTask()

