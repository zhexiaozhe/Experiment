# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 图像后期处理.py
@time: 2017/10/18 11:17
'''
import matplotlib.pyplot as plt
import  matplotlib.image as mpimg
import numpy as np
data=np.load(u'data/ex67.npy')
# D = mpimg.imread('D.png')
# plt.imshow(D)
# plt.show()
plt.plot(data)
plt.show()