# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/15 10:53
'''
import numpy as np
import binascii
import serial
import PyDAQmx

from numpy import pi,sin,cos
from PyDAQmx import *

class CONTROL(object):
    def __init__(self):
        self.ser = serial.Serial('com3', 115200)
        # 使能控制
        self.sgn = lambda x: np.array([0, 1], dtype=np.uint8) if x > 0 else np.array(
                                        [1, 0], dtype=np.uint8) if x < 0 else np.array([0, 0], dtype=np.uint8)
        self.task0 = Task()
        self.task0.CreateDOChan("/Dev2/port0/line0:1", "", PyDAQmx.DAQmx_Val_ChanForAllLines)
        # 输出扭矩控制
        # 扭矩与电压之间的转换关系为：T=1/2.73*value
        self.task1 = Task()
        self.task1.CreateAOVoltageChan("/Dev2/ao0", "", 0, 5, PyDAQmx.DAQmx_Val_Volts, None)
        #角度2采集
        self.sgn2 = lambda x: 1 if x > 0 else -1 if x < 0 else 0
        self.task2 = Task()
        self.read2 = int32()
        self.data2 = numpy.zeros((1,), dtype=numpy.float64)
        self.task2.CreateCICountEdgesChan("Dev2/ctr0", "", DAQmx_Val_Falling, 0, DAQmx_Val_CountUp)
        # 杆2角速度和扭矩采集
        # 角速度与电压之间的转换关系：v=value*pi/3
        self.task3 = Task()
        self.read3 = int32()
        self.data3 = numpy.zeros((2,), dtype=numpy.float64)
        self.task3.CreateAIVoltageChan("Dev2/ai0:1", "", DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)

    def read_ahrs(self):
        b = []
        b1 = []
        # 数据包数据采集
        for _ in range(22):
            a = self.ser.read()
            a = int(binascii.b2a_hex(a).decode('ascii'), 16)
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
        roll = roll * pi / 180
        dr = dr * pi / 180
        return (roll, -dr)

    def start(self):
        # 开启任务
        self.task0.StartTask()
        self.task1.StartTask()
        self.task2.StartTask()
        self.task3.StartTask()
        # 为了计算角度二做准备
        self.angle2 = 0
        self.pre_data2 = np.zeros((1,), dtype=numpy.float64)
        # 给扭矩采集一个初始值，用来纠正偏差
        self.task3.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, self.data3, 2, byref(self.read3), None)
        self.begin_torque = self.data3[1] * 2.73

    def read_daq(self):
        # 杆2角速度采集
        self.task3.ReadAnalogF64(1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, self.data3, 2, byref(self.read3), None)
        self.angle_velocity2 = self.data3[0] *2* pi / 3  # 转换成弧度
        # 杆2角度采集
        self.task2.ReadCounterF64(1, 0.0, self.data2, 1, byref(self.read2), None)
        dir = self.sgn2(self.data3[0])
        dif_data2 = self.data2[0] - self.pre_data2
        dif_angle2 = dir * dif_data2 * (360 / 102400) * pi / 180  # 已经转换成弧度制
        self.angle2 = self.angle2 + dif_angle2
        self.pre_data2 = self.data2[0]
        # 扭矩采集
        self.torque = self.data3[1] * 2.73 - self.begin_torque
        return (self.angle2[0],self.angle_velocity2,self.torque)

    def write_daq(self,value):
        data0 = self.sgn(value)  # 转向使能给定
        self.task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)  # 数字口发送
        self.task1.WriteAnalogScalarF64(1, 10.0, abs(value), None)

    def stop(self):
        data0 = np.array([0, 0], dtype=np.uint8)
        self.task1.WriteAnalogScalarF64(1, 10.0, 0, None)
        self.task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)
        self.task0.StopTask()
        self.task1.StopTask()
        self.task2.StopTask()
        self.task3.StopTask()
        print("停止控制")

    def wrap(self,x, m, M):# 角度角速度处理
        diff = M - m
        while x > M:
            x = x - diff
        while x < m:
            x = x + diff
        return x

    def bound(self,x, m, M=None):
        if M is None:
            M = m[1]
            m = m[0]
        return min(max(x, m), M)

    def state_pre(self,s):
        MAX_VEL_1 = 4 * pi
        MAX_VEL_2 = 9 * pi
        # 角度和角速度处理
        s[0] = self.wrap(s[0], -1.2*pi, pi)
        s[1] = self.wrap(s[1], -pi, pi)
        s[2] = self.bound(s[2], -MAX_VEL_1, MAX_VEL_1)
        s[3] = self.bound(s[3], -MAX_VEL_2, MAX_VEL_2)
        s[2]=s[2]/(4*pi)
        s[3]=s[3]/(9*pi)
        return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])
