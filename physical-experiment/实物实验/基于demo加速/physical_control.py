# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: physical_control.py
@time: 2017/11/19 20:19
'''
# from Virtual_control import *
from Dynamic_Servo import *
import numpy as np
from numpy import pi,sin,cos
import time
import matplotlib.pyplot as plt
from PyDAQmx import *
import serial
import binascii
import PyDAQmx
import tkinter as tk
import threading
##########################################################################

flag=0
# 变量采集
T_collect = []  # 力矩采集
T_send = []
Theta1 = []  # 角度采集
Theta2 = []
Theta1d=[]
Theta2d=[]
Angle_velocity1 = []  # 角速度采集
Angle_velocity2 = []
Angle_v2_smoothing=[]
E=[]
Ed=[]
##########################################################################

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global T_collect,T_send,Theta1,Theta2,Angle_velocity1,Angle_velocity2,Theta1d,Theta2d
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
            b1 = []
            for _ in range(22):
                a = ser.read()
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

            # print("角度：%s 角速度：%s" % (roll, -dr))
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
        # Theta1d,Theta2d=-pi/4,pi/4

        # 给杆2角度计算差值做准备
        angle2 = 0
        pre_data2 = np.zeros((1,), dtype=numpy.float64)

        # 给扭矩采集一个初始值，用来纠正偏差
        task3.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data3, 2, byref(read3), None)
        begin_torque = data3[1] * 2.73

        control=CONTROL()

        for step in range(3000):

            if step==199:
                print('开始采集数据')
                start_time = time.clock()
            # 杆1角度和角速度采集
            # 前面大约200个不知道什么原因
            # 采样时间非常短
            angle1, angle_velocity1 = SER()
            # 杆2角速度采集
            task3.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data3, 2, byref(read3), None)
            angle_velocity2 = data3[0] * pi / 3  # 转换成弧度
            if step==699 or step==0:
                angle_v2_smoothing=angle_velocity2
            angle_v2_smoothing=1.0*angle_velocity2+0.0*angle_v2_smoothing
            # 杆2角度采集
            task2.ReadCounterF64(1, 0.0, data2, 1, byref(read2), None)
            dir = sgn2(data3[0])
            dif_data2 = data2[0] - pre_data2
            dif_angle2 = dir * dif_data2 * (360 / 102400) * pi / 180  # 已经转换成弧度制
            angle2 = angle2 + dif_angle2
            pre_data2 = data2[0]
            #对状态进行预处理和整合
            state=[angle1,angle2,angle_velocity1,angle_velocity2]
            # 发送扭矩
            value = control.Torque(state)
            T=1/2.73*value[0]
            e=value[1]
            e_d=value[2]
            if step>=699:
                if flag==1:
                    T=np.clip(T,-10.8/2.73,10.8/2.73)
                    if step==699:
                        T_smoothing=T
                    T_smoothing=1.0*T+0.0*T_smoothing
                else:
                    T_smoothing=0
            else:
                T_smoothing = 0
            data0 = sgn(T_smoothing)  # 转向使能给定
            task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)  # 数字口发送
            task1.WriteAnalogScalarF64(1, 10.0, abs(T_smoothing), None)  # 模拟口发送
            # 扭矩采集
            # task3.ReadAnalogF64(1, 10.0, DAQmx_Val_GroupByChannel, data3, 2, byref(read3), None)
            torque = data3[1] * 2.73 - begin_torque#data3[1]是上一个点
            # 数据采集
            if step>=199:
                Theta1.append(angle1)
                Theta2.append(angle2)
                Theta1d.append(-pi/4)
                Theta2d.append(pi/4)
                Angle_velocity1.append(angle_velocity1)
                # Angle_v2_smoothing.append(angle_velocity2)
                Angle_velocity2.append(angle_velocity2)
                T_send.append(2.73*T_smoothing)
                E.append(e)
                Ed.append(e_d)
                if step>199:
                    T_collect.append(torque)
        print(time.clock() - start_time)
        data0 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        task0.WriteDigitalLines(1, 1, 10.0, PyDAQmx.DAQmx_Val_GroupByChannel, data0, None, None)

def start_me():
    global flag
    flag=1
    thread=MyThread()
    thread.start()
    print('开始')

def stop_me():
    global flag
    flag=0
    print('停止')

if __name__ == '__main__':
    ###################################################################
    window = tk.Tk()
    window.title('control')
    window.geometry('300x100')

    start = tk.Button(window,
                      text='开始',  # 显示在按钮上的文字
                      width=15, height=2,
                      command=start_me)  # 点击按钮式执行的命令
    start.pack()

    stop = tk.Button(window,
                     text='停止',  # 显示在按钮上的文字
                     width=15, height=2,
                     command=stop_me)  # 点击按钮式执行的命令
    stop.pack()
    window.mainloop()
    ######################################################################
    # 力矩图像
    plt.figure(1)
    plt.title('Action Experiment Figure')
    plt.xlabel('Step/0.02s')
    plt.ylabel('Torque/N.m')
    plt.plot(T_collect, 'r--', label='collect torque')
    plt.plot(T_send, 'b-', label='send torque')
    plt.grid()
    plt.legend()
    # 角度图像
    plt.figure(2)
    plt.title('Angle Experiment Figure')
    plt.xlabel('Step/0.02s')
    plt.ylabel('Angle/rad')
    plt.plot(Theta1, 'r-', label='theta1')
    plt.plot(Theta2, 'b-', label='theta2')
    plt.plot(Theta1d,'y--',label='theta1d')
    plt.plot(Theta2d,'g--',label='theta2d')
    plt.grid()
    plt.legend()
    # 角速度图像
    plt.figure(3)
    plt.title('Angle Velocity Eeperiment Figure')
    plt.xlabel('Step/0.02s')
    plt.ylabel('Angle_velocity/rad/s')
    plt.plot(Angle_velocity1, 'r--', label='Angle_velocity1')
    # plt.plot(Angle_v2_smoothing,'g-',label='Angle_v2_smoothing')
    plt.plot(Angle_velocity2, 'b-', label='Angle_velocity2')
    plt.grid()
    plt.legend()
    plt.figure(4)
    plt.plot(E,label='E')
    plt.plot(Ed,label='Ed')
    plt.legend()
    plt.show()
    ##########################################################################