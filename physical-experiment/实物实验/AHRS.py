import serial
import binascii
from numpy import pi
import time

ser=serial.Serial('com11',115200)
def SER():
    b = []

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
    roll=roll*pi/180
    dr=dr*pi/180

    # print("角度：%s 角速度：%s" % (roll, -dr))
    return (roll,-dr)
start_time=time.clock()
for step in range(1000):
    roll,dr=SER()
    print("角度：%s 角速度：%s" % (roll, dr))
print(time.clock()-start_time)