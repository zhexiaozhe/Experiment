# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/17 16:52
'''

import numpy as np
import random

data=np.load('data/demo_buffer.npy')
print(type(data))
random.sample(list(data),64)
print(len(data))

action=np.asarray([d[1] for d in data])

# print(np.resize(action,[1,1000]))
