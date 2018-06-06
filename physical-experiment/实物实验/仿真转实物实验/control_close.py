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
    angle1d = -np.pi / 8
    angle2d = np.pi / 4
    save_data = SAVE_DATA()
    hand = HAND_SHANK()
    button = BUTTON()
    d = DISTANCE([angle1d,angle2d])
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
    value=0
    t=0
    while True:
        last_time = time.clock()                                                                                        #为了统计每一步运行的时间
        angle1, angle_velocity1 = sen.read_ahrs()
        angle2, angle_velocity2, torque = sen.read_daq()
        state = sen.state_pre([angle1, angle2, angle_velocity1, angle_velocity2])
        D = d.dis([angle1, angle2,angle1d,angle2d])
        button_com = button.state()

        #起摆控制实验
        # if step<1700:
        #     value=0
        # else:
        #     if step==1700:
        #         start=time.clock()
        #     init_value = 10 / 2.73 * agent.action(state)[0]
        #     value = 0.5 * init_value + 0.5 * value
        #     value = 0.5 * init_value + 0.5 * value
        #     t = time.clock() - start
        #     data = [angle1, angle2, angle_velocity1, angle_velocity2, value, torque, t, D, angle1d, angle2d]
        #     save_data.recorde(data)
        # step+=1

        #对称位置抓取控制实验
        if button_com==15:
            if step==0:
                start = time.clock()
            # value=10 / 2.73 * agent.action(state)[0]
            init_value = 10 / 2.73 * agent.action(state)[0]
            value=0.5*init_value+0.5*value
            t = time.clock() - start
            data = [angle1, angle2, angle_velocity1, angle_velocity2, value, torque, t,D,angle1d,angle2d]
            save_data.recorde(data)
            step += 1
        else:
            value=0
        sen.write_daq(value)
        com=hand.command()
        #抓取实验
        # if D < 0.5:
        #     paw.close()
        # if D < 0.05 or step >= 150:
        #     break
        #能量镇定实验
        if com==2 and D<0.6:
            paw.close()
        if com==2 and D<0.05:
            break
        if step>2000:
            break
        #起摆实验
        # if D<0.5 and step>2000:
        #     paw.close()
        #
        # if D<0.1 or step>=2200:
        #     break

        print(value, time.clock() - last_time, step)
    sen.stop()

    print('时间：', time.clock() - start)
    save_data.save_to_file()
    plt.plot(save_data.load())

if __name__=='__main__':
    main()