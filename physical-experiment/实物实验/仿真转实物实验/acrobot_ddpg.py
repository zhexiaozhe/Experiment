import filter_env
from ddpg import *
import tensorflow as tf
import numpy as np
from numpy import pi,sin,cos
import time
import matplotlib.pyplot as plt
from PyDAQmx import *
import serial
import binascii
import PyDAQmx


def main():
    # 实物环境
    # 使能控制
    sgn = lambda x: np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8) if x > 0 else np.array([1, 0, 0, 0, 0, 0, 0, 0],dtype=np.uint8) if x < 0 else np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    task0 = Task()
    task0.CreateDOChan("/Dev2/port0/line0:7", "", PyDAQmx.DAQmx_Val_ChanForAllLines)

    # 输出扭矩控制
    # 扭矩与电压之间的转换关系为：T=1/2.73*value
    task1 = Task()
    task1.CreateAOVoltageChan("/Dev2/ao0", "", 0, 5, PyDAQmx.DAQmx_Val_Volts, None)

    # 杆2角度采集
    sgn2 = lambda x: 1 if x > 0 else -1 if x < 0 else 0
    task2 = Task()
    read2 = int32()
    data2 = numpy.zeros((1,), dtype=numpy.float64)
    task2.CreateCICountEdgesChan("Dev2/ctr0", "", DAQmx_Val_Falling, 0, DAQmx_Val_CountUp)

    # 杆2角速度和扭矩采集
    # 角速度与电压之间的转换关系：v=value*pi/3
    task3 = Task()
    read3 = int32()
    data3 = numpy.zeros((2,), dtype=numpy.float64)
    task3.CreateAIVoltageChan("Dev2/ai0:1", "", DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)

    # 杆1角度和角速度采集
    ser = serial.Serial('com11', 115200)

    def SER():
        b = []
        # 数据包数据采集
        for _ in range(22):
            a = ser.read()
            b.append(a)
        b[9] = int(binascii.b2a_hex(b[9]).decode('ascii'), 16)
        b[10] = int(binascii.b2a_hex(b[10]).decode('ascii'), 16)
        b[18] = int(binascii.b2a_hex(b[18]).decode('ascii'), 16)
        b[19] = int(binascii.b2a_hex(b[19]).decode('ascii'), 16)
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

    #角度角速度处理
    def wrap(x, m, M):
        diff = M - m
        while x > M:
            x = x - diff
        while x < m:
            x = x + diff
        return x

    def bound(x, m, M=None):
        if M is None:
            M = m[1]
            m = m[0]
        return min(max(x, m), M)

    def state_pre(s):
        MAX_VEL_1=4*pi
        MAX_VEL_2=9*pi

        # 角度处理
        s[0] = wrap(s[0], -pi, pi)
        s[1] = wrap(s[1], -pi, pi)
        s[2] = bound(s[2], -MAX_VEL_1, MAX_VEL_1)
        s[3] = bound(s[3], -MAX_VEL_2, MAX_VEL_2)

        return np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3],sin(s[4]), cos(s[4]), sin(s[5]), cos(s[5])])

    task0.StartTask()
    task1.StartTask()
    task2.StartTask()
    task3.StartTask()

    #指定目标点
    Theta1d,Theta2d=-pi/4,pi/2

    # 给杆2角度计算差值做准备
    angle2 = 0
    pre_data2 = np.zeros((1,), dtype=numpy.float64)

    # 给扭矩采集一个初始值，用来纠正偏差
    task3.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data3, 2, byref(read3), None)
    begin_torque = data3[1] * 2.73
    # 变量采集
    T_collect = []  # 力矩采集
    T_send = []
    Theta1 = []  # 角度1采集
    Theta2 = []
    Angle_velocity1 = []  # 角速度采集
    Angle_velocity2 = []
    Time = []

    agent = DDPG()

    for step in range(1000):
        # 杆1角度和角速度采集
        angle1, angle_velocity1 = SER()
        # 杆2角速度采集
        task3.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data3, 2, byref(read3), None)
        angle_velocity2 = data3[0] * pi / 3  # 转换成弧度
        # 杆2角度采集
        task2.ReadCounterF64(1, 0.0, data2, 1, byref(read2), None)
        dir = sgn2(data3[0])
        dif_data2 = data2[0] - pre_data2
        dif_angle2 = dir * dif_data2 * (360 / 102400) * pi / 180  # 已经转换成弧度制
        angle2 = angle2 + dif_angle2
        pre_data2 = data2[0]
        #对状态进行预处理和整合
        state=state_pre([angle1,angle2,angle_velocity1,angle_velocity2,Theta1d,Theta2d])
        # 发送扭矩
        # value=0
        if step<200:
            value=6/2.73*sin(0.04*pi*step)
        else:
            value = 10/2.73*agent.action(state)[0]
        print(value)
        data0 = sgn(value)  # 转向使能给定
        task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)  # 数字口发送
        task1.WriteAnalogScalarF64(1, 10.0, abs(value), None)  # 模拟口发送
        # 扭矩采集
        torque = data3[1] * 2.73 - begin_torque
        time.sleep(0.02)


if __name__ == '__main__':
    main()