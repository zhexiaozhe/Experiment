# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2017/12/23 18:47
'''
import serial
from numpy import pi
import binascii
import sys

class READ_AHRS():
    def __init__(self):
        try:
            self.ser=ser=serial.Serial('com11',115200)
            print("姿态位移传感器连接成功")
        except:
            print("姿态位移传感器连接失败")
            sys.exit()
    def read(self):
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