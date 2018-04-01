# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/3/31 9:18
'''
import time
import numpy as np

from sensor import CONTROL
from hand_shank import HAND_SHANK
from button_input import BUTTON
from save_data import SAVE_DATA
from distance import DISTANCE
from plotting import PLOT

def main():
    save_data = SAVE_DATA()
    hand = HAND_SHANK()
    button=BUTTON()
    d = DISTANCE()
    plt = PLOT()
    sen = CONTROL()
    sen.start()

    Action = np.load(r'data\torque.npy')
    action_len=len(Action)

    step = 0

    while True:
        com=hand.command()
        if com==1:
            break

    while True:
        last_time=time.clock()
        angle1, angle_velocity1 = sen.read_ahrs()
        angle2, angle_velocity2, torque = sen.read_daq()
        D = d.dis([angle1, angle2])
        com=button.state()
        if com==15:
            if step==0:
                start = time.clock()
            value=10 / 2.73 * Action[step]
            t=time.clock() - start
            data = [angle1, angle2, angle_velocity1, angle_velocity2, value, torque, t]
            save_data.recorde(data)
            step+=1
        else:
            value=0
        sen.write_daq(value)
        print(value,time.clock()-last_time)
        if step>=action_len:
            break
    sen.stop()
    print(time.clock() - start)
    save_data.save_to_file()
    plt.plot(save_data.load())

if __name__ =='__main__':
    main()