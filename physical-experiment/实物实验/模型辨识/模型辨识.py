# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2017/12/23 18:22
'''
from numpy import pi,sin
from read_ahrs import READ_AHRS
from daq_card import DAQ
from save_data import SAVE_DATA
from plotting import PLOT
from hand_shank import HAND_SHANK
import time

def main():
    SER=READ_AHRS()
    daq=DAQ()
    daq.start()
    save_data=SAVE_DATA()
    start_time = time.clock()
    plt=PLOT()
    hand=HAND_SHANK()
    for step in range(1000):
        if step < 200:
            value = 0
        else:
            value = 6/2.73 * sin(0.02*pi * (step-200))
        value=hand.control()
        daq.write_data(value)
        angle1,angle_velocity1=SER.read()
        angle2,angle_velocity2,toqure=daq.read_data()
        t=time.clock() - start_time
        data=[angle1,angle2,angle_velocity1,angle_velocity2,value,toqure,t]
        save_data.recorde(data)
    daq.stop()
    save_data.save_to_file()
    plt.plot(save_data.load())
if __name__ =='__main__':
    main()