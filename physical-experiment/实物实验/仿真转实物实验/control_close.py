# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/4/1 9:38
'''

import time
import numpy as np

from sensor import CONTROL
from hand_shank import HAND_SHANK
from button_input import BUTTON
from save_data import SAVE_DATA
from distance import DISTANCE
from paw_control import PAW
from plotting import PLOT
from ddpg import DDPG

def main():
    save_data = SAVE_DATA()
    hand = HAND_SHANK()
    button = BUTTON()
    d = DISTANCE()
    plt = PLOT()
    sen = CONTROL()
    sen.start()
    agent=DDPG()
    paw=PAW()
    paw.open()

    step = 0

    while True:
        com=hand.command()
        if com==1:
            break

    while True:
        last_time = time.clock()                                                                                        #为了统计每一步运行的时间
        angle1, angle_velocity1 = sen.read_ahrs()
        angle2, angle_velocity2, torque = sen.read_daq()
        state = sen.state_pre([angle1, angle2, angle_velocity1, angle_velocity2])
        D = d.dis([angle1, angle2])
        button_com = button.state()
        if button_com==15:
            if step==0:
                start = time.clock()
            value = 10 / 2.73 * agent.action(state)[0]
            t = time.clock() - start
            data = [angle1, angle2, angle_velocity1, angle_velocity2, value, torque, t]
            save_data.recorde(data)
            step += 1
        else:
            value=0
        sen.write_daq(value)

        if D<0.2 or step>=100:
            paw.close()
            break

        print(value, time.clock() - last_time, step)
    sen.stop()

    print('时间：', time.clock() - start)
    save_data.save_to_file()
    plt.plot(save_data.load())

if __name__=='__main__':
    main()