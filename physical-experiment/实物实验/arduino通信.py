# encoding: utf-8
'''
@author: 程哲
@contact: 909991719@qq.com
@file: arduino通信.py
@time: 2017/9/28 10:55
'''
import serial
ser=serial.Serial("com8",9600)
ser.timeout=1
print(ser.readline())
ser.write(b'11111\n')
print(ser.readline().decode("ascii"))
ser.close()