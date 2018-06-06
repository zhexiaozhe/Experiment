# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2017/12/23 20:35
'''

import matplotlib.pyplot as plt

from save_data import SAVE_DATA

class PLOT():
    def __init__(self):
        pass
    def plot(self,data):
        Theta1=data[0]
        Theta2=data[1]
        Angle_velocity1=data[2]
        Angle_velocity2=data[3]
        T_send = data[4]
        T_collect =data[5]
        Distance=data[6]
        Theta1d=data[7]
        Theta2d=data[8]

        plt.figure('动作曲线')
        plt.title('Action Experiment')
        plt.xlabel('Step/0.02s')
        plt.ylabel('Torque/N.m')
        plt.plot(T_collect[1:], 'r--', label='collect torque')
        plt.plot(T_send, 'b-', label='send torque')
        plt.grid()
        plt.legend()

        plt.figure('角度曲线')
        # plt.title('Angle Experiment')
        plt.xlabel('time/s')
        plt.ylabel('Angle/rad')
        plt.plot(Theta1, 'r--', label='theta1')
        plt.plot(Theta2, 'b-', label='theta2')
        plt.plot(Theta1d,'g-.',label='theta1d')
        plt.plot(Theta2d,'c:',label='theta2d')
        plt.grid()
        plt.legend()

        plt.figure('角速度曲线')
        # plt.title('Angle Velocity Experiment')
        plt.xlabel('Step/0.02s')
        plt.ylabel('Angle_velocity/rad/s')
        plt.plot(Angle_velocity1, 'r--', label='Angle_velocity1')
        plt.plot(Angle_velocity2, 'b-', label='Angle_velocity2')
        plt.grid()
        plt.legend()

        plt.figure('距离曲线')
        plt.title('Distance Experiment')
        plt.xlabel('Step/0.02s')
        plt.ylabel('distance/m')
        plt.plot(Distance, 'b-', label='distance')
        plt.grid()
        plt.legend()
        plt.show()


if __name__ =='__main__':
    plot = PLOT()
    save_data = SAVE_DATA()
    plot.plot(save_data.load())