# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/5/10 9:57
'''
import serial
import binascii
from numpy import pi
import time
from AHRS import AHRS_OLD
import matplotlib.pyplot as plt

class AHRS_NEW(object):
    def __init__(self):
        self.ser=serial.Serial('com4',115200)
    def read(self):
        b = []
        b1=[]
        for _ in range(22):

            a = self.ser.read()
            a=int(binascii.b2a_hex(a).decode('ascii'), 16)
            b1.append(a)
    #增加判断机制
        for i in range(22):
            m=i
            if b1[i]==90 and b1[i+1]==165:
                break
        j=0
        for i in range(22):
            if m+i<22:
                b.append(b1[m+i])
            else:
                b.append(b1[j])
                j+=1

        # 角度计算
        if b[19] < 127:
            roll = (b[19] * 256 + b[18]) / 100
        else:
            roll = ((b[19] - 256) * 256 + b[18]) / 100
            # 角速度计算
        if b[10] < 127:
            dr = (b[10] * 256 + b[9]) * 0.1
        else:
            dr = ((b[10] - 256) * 256 + b[9]) * 0.1
            # 转换成弧度制
        roll=roll*pi/180
        dr=dr*pi/180

        # print("角度：%s 角速度：%s" % (roll, -dr))
        return (roll,dr)

if __name__=='__main__':
    start_time = time.clock()
    ahrs_old=AHRS_OLD()
    ahrs_new = AHRS_NEW()
    Angle1=[]
    Theta1=[]
    Angle2 = []
    Theta2 = []
    for step in range(1000):
        roll1,dr1=ahrs_old.read()
        roll,dr=ahrs_new.read()
        # print("角度1：%s 角速度1：%s 步数：%s" % (roll1, dr1, step))
        print("角度：%s 角速度：%s 步数：%s" % (roll, dr, step))
        Angle1.append(roll)
        Theta1.append(dr)
        Angle2.append(roll1)
        Theta2.append(dr1)
    plt.figure(1)
    plt.plot(Angle1,label='new')
    plt.plot(Angle2,label='old')
    plt.legend()

    plt.figure(2)
    plt.plot(Theta1,label='new')
    plt.plot(Theta2,label='old')
    plt.legend()
    plt.show()
    print(time.clock() - start_time)