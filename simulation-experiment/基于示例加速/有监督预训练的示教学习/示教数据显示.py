# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/4/23 11:19
'''

import numpy as np
import pickle

with open('data\object_demo.pickle', 'rb') as f:
    buffer = pickle.load(f)
print(buffer[0])
print(np.shape(buffer))