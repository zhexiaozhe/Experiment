# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/3/28 16:10
'''

import math
import numpy as np

class DISTANCE(object):
    def __init__(self,data):
        self.LINK_LENGTH_1=0.593
        self.LINK_LENGTH_2 = 0.593
        self.theta1d=data[0]
        self.theta2d=data[1]

    def dis(self,s):
        # 模型改目标的位置
        g_p1 = [self.LINK_LENGTH_1 *
                np.sin(self.theta1d), self.LINK_LENGTH_1 * np.cos(self.theta1d)]

        g_p2 = [g_p1[0] + self.LINK_LENGTH_2 * np.sin(self.theta1d + self.theta2d),
                g_p1[1] + self.LINK_LENGTH_2 * np.cos(self.theta1d + self.theta2d)]
        # 模型改末端位置
        p1 = [self.LINK_LENGTH_1 *
              np.sin(s[0]), self.LINK_LENGTH_1 * np.cos(s[0])]

        p2 = [p1[0] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.cos(s[0] + s[1])]
        # 末端与目标点之间的距离
        D = math.sqrt((p2[0] - g_p2[0]) ** 2 + (p2[1] - g_p2[1]) ** 2)
        return D