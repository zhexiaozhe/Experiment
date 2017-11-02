# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 模型参数辨识.py
@time: 2017/10/8 9:04
'''

import gym
import numpy as np
from numpy import pi,sin,cos
import time
import matplotlib.pyplot as plt
from PyDAQmx import *
import serial
import binascii
import PyDAQmx
import scipy.io as sio

#实物环境
#使能控制
sgn=lambda x:np.array([0,1,0,0,0,0,0,0], dtype=np.uint8) if x>0 else np.array([1,0,0,0,0,0,0,0], dtype=np.uint8)  if x<0 else np.array([0,0,0,0,0,0,0,0], dtype=np.uint8)
task0 = Task()
task0.CreateDOChan("/Dev2/port0/line0:7","",PyDAQmx.DAQmx_Val_ChanForAllLines)

#输出扭矩控制
#扭矩与电压之间的转换关系为：T=1/2.73*value
task1=Task()
task1.CreateAOVoltageChan("/Dev2/ao0","",0,5,PyDAQmx.DAQmx_Val_Volts,None)

#杆2角度采集
sgn2=lambda x:1 if x>0 else -1 if x<0 else 0
task2=Task()
read2 = int32()
data2 = numpy.zeros((1,), dtype=numpy.float64)
task2.CreateCICountEdgesChan("Dev2/ctr0","",DAQmx_Val_Falling,0,DAQmx_Val_CountUp)

#杆2角速度和扭矩采集
#角速度与电压之间的转换关系：v=value*pi/3
task3=Task()
read3 = int32()
data3 = numpy.zeros((2,), dtype=numpy.float64)
task3.CreateAIVoltageChan("Dev2/ai0:1","",DAQmx_Val_Cfg_Default,-10.0,10.0,DAQmx_Val_Volts,None)

#杆1角度和角速度采集
ser=serial.Serial('com11',115200)
def SER():
    b = []
    b1=[]
    #数据包数据采集
    for _ in range(22):
        a = ser.read()
        a=int(binascii.b2a_hex(a).decode('ascii'), 16)
        b1.append(a)
        # 增加判断机制
    for i in range(22):
        m = i
        if b1[i] == 90 and b1[i + 1] == 165:
            break
    j = 0
    for i in range(22):
        if m + i < 22:
            b.append(b1[m + i])
        else:
            b.append(b1[j])
            j += 1
    # 角度计算
    if b[19] < 127:
        roll = (b[19] * 256 + b[18]) / 100
    else:
        roll = ((b[19] - 256) * 256 + b[18]) / 100
    # 角速度计算
    if b[10] < 127:
        dr = (b[10] * 256 + b[9]) * (2000 / 32768)
    else:
        dr = ((b[10] - 256) * 256 + b[9]) * (2000 / 32768)
    # 转换成弧度制
    roll=roll*pi/180
    dr=dr*pi/180
    return (roll,-dr)

task0.StartTask()
task1.StartTask()
task2.StartTask()
task3.StartTask()

#给杆2角度计算差值做准备
angle2=0
pre_data2=np.zeros((1,), dtype=numpy.float64)
#给定力矩
dataFile = u'F:/matlab_file/xiaogu.mat'
data = sio.loadmat(dataFile)
t=data['torque01']
#给扭矩采集一个初始值，用来纠正偏差
task3.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data3, 2, byref(read3), None)
begin_torque=data3[1]*2.73
#变量采集
T_collect=[] #力矩采集
T_send=[]
Theta1=[] #角度1采集
Theta2=[]
Angle_velocity1=[] #角速度采集
Angle_velocity2=[]
Time=[]

start_time=time.clock()
for step in range(501):
    #定义扭矩
    # value=6/2.73 * np.sin(0.02*pi * step)
    # value=t[step]/2.73
    if step<100:
        value = 0
    else:
        value = 6/2.73 * np.sin(0.02*pi * (step-100))
    data0=sgn(value)#转向使能给定
    task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)#数字口发送
    task1.WriteAnalogScalarF64(1, 10.0, abs(value), None)#模拟口发送
    #杆1角度和角速度采集
    angle1,angle_velocity1=SER()
    #杆2角速度采集
    task3.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data3, 2, byref(read3), None)
    angle_velocity2 = data3[0] * pi / 3 #转换成弧度
    #杆2角度采集
    task2.ReadCounterF64(1, 0.0, data2, 1, byref(read2), None)
    dir = sgn2(data3[0])
    dif_data2 = data2[0] - pre_data2
    dif_angle2 = dir * dif_data2 * (360 / 102400)*pi/180#已经转换成弧度制
    angle2 = angle2 + dif_angle2
    pre_data2 = data2[0]
    #发送扭矩
    task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)  # 数字口发送
    task1.WriteAnalogScalarF64(1, 10.0, abs(value), None)  # 模拟口发送
    # 扭矩采集
    torque = data3[1] * 2.73 - begin_torque
    #时间采集
    Time.append(time.clock() - start_time - 0.018)

    T_collect.append(torque)
    T_send.append(2.73*value)
    Theta1.append(angle1)
    Theta2.append(angle2[0])
    Angle_velocity1.append(angle_velocity1)
    Angle_velocity2.append(angle_velocity2)
    print("角度1：",angle1," 角度2：",angle2[0]," 角速度1: ",angle_velocity1," 角速度2: ",angle_velocity2)
    # time.sleep(0.01)
print(time.clock()-start_time)
#采集完成恢复竖直位置
data0=np.array([0,0,0,0,0,0,0,0], dtype=np.uint8)
task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)
task0.StopTask()
task1.StopTask()
task2.StopTask()
task3.StopTask()
#数据保存
np.save('data\Theta1.npy',np.array(Theta1))
np.save('data\Theta2.npy',np.array(Theta2))
np.save('data\Torque.npy',np.array(T_send))
np.save('data\Time.npy',np.array(Time))

#生成图像
plt.figure(1)
plt.title('Action Experiment Figure')
plt.xlabel('Step/0.02s')
plt.ylabel('Torque/N.m')
plt.plot(T_collect,'r--',label='collect torque')
plt.plot(T_send,'b-',label='send torque')
plt.grid()
plt.legend()

plt.figure(2)
plt.title('Angle Experiment Figure')
plt.xlabel('Step/0.02s')
plt.ylabel('Angle/rad')
plt.plot(Theta1,'r--',label='theta1')
plt.plot(Theta2,'b-',label='theta2')
plt.grid()
plt.legend()

plt.figure(3)
plt.title('Angle Velocity Eeperiment Figure')
plt.xlabel('Step/0.02s')
plt.ylabel('Angle_velocity/rad/s')
plt.plot(Angle_velocity1,'r--',label='Angle_velocity1')
plt.plot(Angle_velocity2,'b-',label='Angle_velocity2')
plt.grid()
plt.legend()

plt.show()


