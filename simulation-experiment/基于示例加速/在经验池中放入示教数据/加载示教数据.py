# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/5/15 10:54
'''
import numpy as np
import pickle
from collections import deque

with open('object.pickle', 'rb') as f:
    buffer=pickle.load(f)
print(np.shape(buffer))
# for i in range(len(buffer)):
#     if(buffer[i][4]==1):
#         print(buffer[i],i)
#         buffer_good.append(buffer[i])
#         print(j)
#         j+=1
