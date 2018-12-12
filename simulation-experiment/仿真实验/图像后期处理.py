# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 图像后期处理.py
@time: 2017/10/18 11:17
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.io as sio
# data=np.load(u'data/ex67.npy')
# D = mpimg.imread('D.png')
# plt.imshow(D)
# plt.show()
# plt.plot(data)
# plt.show()

# x=np.arange(0,4100,100)

dataFile = u'F:/matlab_files/中期实验3.mat'
y=np.load('data\中期实验4.npy')
# theta1=np.load('data\mid\Theta1.npy')
# theta2=np.load('data\mid\Theta2.npy')
# theta1d=np.load('data\mid\Theta1d.npy')
# theta2d=np.load('data\mid\Theta2d.npy')
# sio.savemat(dataFile, {'Theta1':theta1,'Theta2':theta2,'Theta2d':theta2d,'Theta1d':theta1d})
# sio.savemat(dataFile, {'Y':y})
# plt.plot(x,y[0:41])
plt.plot(y)
# plt.grid()
plt.show()