import serial
import binascii
from numpy import pi
import time
import numpy as np
ser=serial.Serial('com11',115200)
def SER():
    b = []
    b1=[]
    for _ in range(22):

        a = ser.read()
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
        dr = (b[10] * 256 + b[9]) * (2000 / 32768)
    else:
        dr = ((b[10] - 256) * 256 + b[9]) * (2000 / 32768)
        # 转换成弧度制
    roll=roll*pi/180
    dr=dr*pi/180

    # print("角度：%s 角速度：%s" % (roll, -dr))
    return (roll,-dr)
start_time = time.clock()
for step in range(10000000):

    roll,dr=SER()

    print("角度：%s 角速度：%s 步数：%s" % (roll, dr, step))
print(time.clock() - start_time)