# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: py2mat数据.py
@time: 2017/9/30 12:50
'''

import numpy as np
import scipy.io as sio

#matlab数据读取
dataFile = u'F:/matlab_files/实物实验3.mat'
# data = sio.loadmat(dataFile)
# print(type(data))
# print(data['ans'])

#numpy数据读取

a0=np.load("data\Theta1.npy")
a1=np.load("data\Theta2.npy")
a2=np.load("data\Send_Torque.npy")
a3=np.load("data\distance.npy")
a4=np.load("data\Time.npy")
a5=np.load("data\Theta1d.npy")
a6=np.load("data\Theta2d.npy")
#将numpy数据保存成matlab数据
sio.savemat(dataFile, {'Theta1':a0,'Theta2':a1,'Send_Torque':a2,'Distance':a3,'time':a4,'Theta1d':a5,'Theta2d':a6})

#仿真数据发送到matlab
# s_a0=np.load("data\S_Theta1.npy")
# s_a1=np.load("data\S_Theta2.npy")
# s_a2=np.load("data\S_torque.npy")
# sio.savemat(dataFile, {'S_Theta1':s_a0,'S_Theta2':s_a1,'S_Torque':s_a2})