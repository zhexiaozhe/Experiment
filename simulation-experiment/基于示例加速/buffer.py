# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/17 18:32
'''

from collections import deque
import numpy as np
import random

class ReplayDemo(object):
    def __init__(self):
        self.buffer=list(np.load('data/demo_buffer.npy'))

    def get_batch(self,batch_size):
        return random.sample(self.buffer,batch_size)

    def size(self):
        return len(self.buffer)

class ReplayBuffer(object):
    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0