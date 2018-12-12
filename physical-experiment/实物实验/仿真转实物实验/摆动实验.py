# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/11/20 9:31
'''
from numpy import pi,sin
from ddpg import DDPG
from plotting import *
from sensor import CONTROL
from hand_shank import HAND_SHANK
from distance import DISTANCE
from paw_control import PAW
from save_data import SAVE_DATA
from plotting import PLOT

def main():
    angle1d = -pi / 4
    angle2d = pi / 2
    save_data = SAVE_DATA()
    hand = HAND_SHANK()
    sen = CONTROL()
    sen.start()
    agent = DDPG()
    plt = PLOT()
    paw = PAW()
    paw.open()
    d = DISTANCE([angle1d, angle2d])
    step=0
    while True:
        angle1, angle_velocity1 = sen.read_ahrs()
        angle2, angle_velocity2, torque = sen.read_daq()
        D = d.dis([angle1, angle2, angle1d, angle2d])
        state=hand.command()
        if state==0:
            print('等待开始')
            continue
        elif state==2 or angle2>5*pi/6 or angle2<-5*pi/6:
            print('停止')
            sen.stop()
            break
        # elif D<0.05:
        #     print('到达')
        #     paw.close()
        #     sen.stop()
        #     break
        elif state==1:
            print('开始')
            state = sen.state_pre([angle1, angle2, angle_velocity1, angle_velocity2])
            #稀疏奖励
            if step<700:
                action=4.7/2.73 * sin(0.01*pi * step)
                # action=10/2.73
            else:
                action = 10 / 2.73 * agent.action(state)[0]
                if D<0.05:
                    sen.stop()
                    break
                elif D<0.4:
                    paw.close()
            #基于能量
            # if step < 700:
            #     action=7/2.73 * sin(0.01*pi * step)
            # elif step>3000:
            #     action = 10 / 2.73 * agent.action(state)[0]
            #     if D<0.05:
            #         sen.stop()
            #         break
            #     elif D<0.55:
            #         paw.close()
            # else:
            #     action = 10 / 2.73 * agent.action(state)[0]
            data = [angle1, angle2, angle_velocity1, angle_velocity2, action, torque, D,angle1d, angle2d,step]
            save_data.recorde(data)
            # action = 10 / 2.73 * agent.action(state)[0]
            sen.write_daq(action)
            step = step + 1
    save_data.save_to_file()
    plt.plot(save_data.load())

if __name__=='__main__':
    main()