# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/4/30 9:11
'''

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#增加中文功能
plt.rcParams['axes.unicode_minus']=False

# test_data=np.load(u'data\mid\中期实验2.npy')
# step=[50*j for j in range(81)]
# plt.figure('测试图')
# plt.plot(step,test_data,label='loss')
# plt.grid()
# plt.xlabel('episode')
# plt.ylabel('loss')
# plt.show()

dataFile = u'F:/matlab_files/PER实验13.mat'

a0=np.load("data\PER实验13.npy")
# a1=np.load("data\Theta2.npy")
# a3=np.load("data\Time.npy")

sio.savemat(dataFile, {'value13':a0})