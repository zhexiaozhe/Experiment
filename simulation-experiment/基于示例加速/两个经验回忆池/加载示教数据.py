# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/5/15 10:54
'''
import numpy as np
import pickle
from collections import deque

buffer=deque()
print(np.size(buffer,0))
with open('data\毕业实验12.pickle', 'rb') as f:
    buffer=pickle.load(f)
print(np.shape(buffer))
print(buffer[0])
print(np.size(buffer,0))
j=0
for i in range(len(buffer)):
    # print(buffer[i][0])
    if(buffer[i][4]==1):
        print(buffer[i],i)
        # buffer_good.append(buffer[i])
        print(j)
        j+=1
print(i)