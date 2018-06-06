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
# data=np.load(u'data/ex67.npy')

# D = mpimg.imread('D.png')
# plt.imshow(D)
# plt.show()
# plt.plot(data)
# plt.show()
m=0
y=np.load('data\示教实验6.npy')
print()
print(np.shape(y))
x=np.zeros((201,2))
print(np.shape(x))
for i in range(2):
    n=0
    for j in range(201):
        for k in range(20):
            m=np.random.random()
        x[n][i]=m
        n+=1
plt.figure(1)
plt.plot(y)
plt.show()