from collections import deque #双向排列的库
import random
import numpy as np
import time
import pickle

class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    # 当有新 sample 时, 添加进 tree 和 data
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = int(0.2*self.capacity)

    # 当 sample 被 train, 有了新的 TD-error, 就在 tree 中更新
    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree 改变树的结构
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2 #取整除，返回商的整数部分
            self.tree[tree_idx] += change #由最下面的树叶端开始向上更新

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class PrioritizedReplayBuffer(object):

    epsilon = 0.01  # small amount to avoid zero priority 一个小数避免零的优先级
    alpha = 0.6  # [0~1] convert the importance of TD error to priority 转换TD误差为优先级
    beta = 0.4  # importance-sampling, from initial value increasing to 1 重要性采样
    beta_increment_per_sampling = 0.00001  # beta的每次采样后增加的数
    abs_err_upper = 1.  # clipped abs error 绝对误差上限

    def __init__(self, buffer_size,demo):

        self.buffer_size = buffer_size
        self.tree = SumTree(self.buffer_size)
        self.num_experiences = 0
        if demo:
            print('使用优先采用并加载数据实验')

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        self.num_experiences += 1
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, experience)  # set the max p for new p 设置最大的概率给新的数据

    def sample(self, n):
        self.b_idx, self.b_memory, self.ISWeights,pri= np.empty((n,), dtype=np.int32), np.zeros(n, dtype=object), np.empty((n, 1)),np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1) #确定每个区间的上下限
            v = np.random.uniform(a, b) #在区间内进行随机选数
            idx, p, data = self.tree.get_leaf(v) #获取数据
            pri[i, 0] = p
            prob = p / self.tree.total_p
            self.ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            self.b_idx[i], self.b_memory[i] = idx, data

        return self.b_idx, self.b_memory, self.ISWeights,pri

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.num_experiences = 0

# 原来的代码，没有优先采用
class ReplayBuffer(object):
    def __init__(self, buffer_size,demo):

        self.buffer_size = buffer_size
        self.demo=demo
        if self.demo:
            with open('data\object.pickle', 'rb') as f:
                self.buffer=pickle.load(f)
                print("数据加载完成")
        else:
            self.buffer = deque()
        self.num_experiences = np.size(self.buffer,0)

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
            self.num_experiences += 1
    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0