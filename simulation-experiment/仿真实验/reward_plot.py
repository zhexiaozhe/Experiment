# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: reward_plot.py
@time: 2017/11/3 9:10
'''
import matplotlib.pyplot as plt
import numpy as np

data=np.load('data\ex62.npy')
plt.plot(data)
plt.show()