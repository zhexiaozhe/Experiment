# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 扭矩采集.py
@time: 2017/10/2 18:01
'''
from PyDAQmx import *
import time
read=int32()
data=numpy.zeros((1,),dtype=numpy.float64)
task=Task()
task.CreateAIVoltageChan("Dev2/ai1","",DAQmx_Val_Cfg_Default,-4.0,4.0,DAQmx_Val_Volts,None)
task.StartTask()
#采集起始的偏差
task.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data, 1, byref(read), None)
begin_torque=data*2.73
while 1:
    #扭矩采集
    task.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data, 1, byref(read), None)
    print(data)
    torque=data*2.73-begin_torque
    # print("电机的扭矩为：%f N·m"%torque)
    time.sleep(0.01)
task.StopTask()