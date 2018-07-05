from collections import deque #双向排列的库
import random
import numpy as np
import time
import pickle

class ReplayBuffer(object):
    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        # with open('data\object.pickle', 'rb') as f:
        #     self.buffer=pickle.load(f)
        #     print("数据加载完成")
        #     print(np.shape(self.buffer))

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