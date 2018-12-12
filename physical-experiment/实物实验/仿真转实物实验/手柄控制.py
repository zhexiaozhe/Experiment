# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/11/18 19:58
'''

import numpy as np
from sensor import CONTROL
from hand_shank import HAND_SHANK

hand = HAND_SHANK()
sen = CONTROL()
sen.start()
while True:
    angle1, angle_velocity1 = sen.read_ahrs()
    angle2, angle_velocity2, torque = sen.read_daq()
    print(angle2,'   ',angle_velocity2)
    state = sen.state_pre([angle1, angle2, angle_velocity1, angle_velocity2])
    action=hand.control
    print(action)
    sen.write_daq(action)
