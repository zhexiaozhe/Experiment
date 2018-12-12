# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2017/12/23 18:22
'''
import time

from numpy import pi,sin
from sensor import CONTROL
from save_data import SAVE_DATA
from plotting import PLOT
from hand_shank import HAND_SHANK

def main():
    sen = CONTROL()
    sen.start()
    save_data=SAVE_DATA()
    start_time = time.clock()
    plt=PLOT()
    hand=HAND_SHANK()
    for step in range(1000):
        if step < 200:
            value = 0
        else:
            value = 7/2.73 * sin(0.01*pi * (step-200))
        # value=hand.control
        sen.write_daq(value)
        angle1, angle_velocity1 = sen.read_ahrs()
        angle2, angle_velocity2, torque = sen.read_daq()
        t=time.clock() - start_time
        data=[angle1,angle2,angle_velocity1,angle_velocity2,value,torque,t]
        save_data.recorde(data)
    sen.stop()
    save_data.save_to_file()
    plt.plot(save_data.load())

if __name__ =='__main__':
    main()