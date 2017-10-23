
# encoding: utf-8
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 数据采集卡.py
@time: 2017/9/25 10:32
'''

from PyDAQmx import Task
import PyDAQmx
import numpy as np
import time

#电压值与扭矩值之间的转换关系
# T=1/2.73*value
sgn=lambda x:np.array([1,0,0,0,0,0,0,0], dtype=np.uint8) if x>0 else np.array([0,1,0,0,0,0,0,0], dtype=np.uint8)  if x<0 else np.array([0,0,0,0,0,0,0,0], dtype=np.uint8)
value =2
data = np.array([0,0,0,0,0,0,0,0], dtype=np.uint8)
task = Task()
task1=Task()
task1.CreateAOVoltageChan("/Dev2/ao0","",0,5,PyDAQmx.DAQmx_Val_Volts,None)
task.CreateDOChan("/Dev2/port0/line0:7","",PyDAQmx.DAQmx_Val_ChanForAllLines)
task.StartTask()
task1.StartTask()
# task.WriteDigitalLines(1,1,10.0,PyDAQmx.DAQmx_Val_GroupByChannel,data,None,None)
start_time=time.clock()
for i in range(10000):
    # value=2.5*np.sin(0.01*i)
    data=sgn(value)
    # print(value,data)
    task.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data, None, None)
    task1.WriteAnalogScalarF64(1,10.0,abs(value),None)

    time.sleep(0.01)
print(time.clock()-start_time)
task.StopTask()
task1.StopTask()
