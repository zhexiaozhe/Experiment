# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2017/12/23 18:39
'''
import PyDAQmx
from PyDAQmx import *
import numpy as np
from numpy import pi,sin,cos
import sys

class DAQ():
    def __init__(self):
        try:
            #使能控制
            self.sgn = lambda x: np.array([0, 1], dtype=np.uint8) if x > 0 else np.array(
                                         [1, 0],dtype=np.uint8) if x < 0 else np.array([0, 0], dtype=np.uint8)
            self.task0 = Task()
            self.task0.CreateDOChan("/Dev2/port0/line0:1", "", PyDAQmx.DAQmx_Val_ChanForAllLines)
            #模拟量输出
            self.task1 = Task()
            self.task1.CreateAOVoltageChan("/Dev2/ao0", "", 0, 5, PyDAQmx.DAQmx_Val_Volts, None)
            #驱动关节角度
            self.sgn2 = lambda x: 1 if x > 0 else -1 if x < 0 else 0
            self.task2 = Task()
            self.read2 = int32()
            self.data2 = numpy.zeros((1,), dtype=numpy.float64)
            self.task2.CreateCICountEdgesChan("Dev2/ctr0", "", DAQmx_Val_Falling, 0, DAQmx_Val_CountUp)
            #驱动关节角速度
            self.task3 = Task()
            self.read3 = int32()
            self.data3 = numpy.zeros((2,), dtype=numpy.float64)
            self.task3.CreateAIVoltageChan("Dev2/ai0:1", "", DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
            print('采集卡连接成功')
        except:
            print('采集卡连接失败')
            sys.exit()
    def start(self):
        #开启任务
        self.task0.StartTask()
        self.task1.StartTask()
        self.task2.StartTask()
        self.task3.StartTask()
        #为了计算角度二做准备
        self.angle2 = 0
        self.pre_data2 = np.zeros((1,), dtype=numpy.float64)
        # 给扭矩采集一个初始值，用来纠正偏差
        self.task3.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, self.data3, 2, byref(self.read3), None)
        self.begin_torque = self.data3[1] * 2.73

    def read_data(self):
        # 杆2角速度采集
        self.task3.ReadAnalogF64(1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, self.data3, 2, byref(self.read3), None)
        self.angle_velocity2 = self.data3[0] * pi / 3  # 转换成弧度
        #杆2角度采集
        self.task2.ReadCounterF64(1, 0.0, self.data2, 1, byref(self.read2), None)
        dir = self.sgn2(self.data3[0])
        dif_data2 = self.data2[0] - self.pre_data2
        dif_angle2 = dir * dif_data2 * (360 / 102400) * pi / 180  # 已经转换成弧度制
        self.angle2 = self.angle2 + dif_angle2
        self.pre_data2 = self.data2[0]
        # 扭矩采集
        self.torque = self.data3[1] * 2.73 - self.begin_torque
        return (self.angle2[0],self.angle_velocity2,self.torque)

    def write_data(self,value):
        data0 = self.sgn(value)  # 转向使能给定
        self.task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)  # 数字口发送
        self.task1.WriteAnalogScalarF64(1, 10.0, abs(value), None)

    def stop(self):
        data0 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.task1.WriteAnalogScalarF64(1, 10.0, 0, None)
        self.task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)
        self.task0.StopTask()
        self.task1.StopTask()
        self.task2.StopTask()
        self.task3.StopTask()
        print("回到初始位置")