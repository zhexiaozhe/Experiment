# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/15 10:36
'''
import numpy as np

class SAVE_DATA(object):
    def __init__(self):
        self.T_collect = []
        self.T_send = []
        self.Theta1 = []
        self.Theta2 = []
        self.Angle_velocity1 = []
        self.Angle_velocity2 = []

    def recorde(self,data):
        Theta1=data[0]
        Theta2=data[1]
        dtheta1=data[2]
        dtheta2=data[3]
        T_send = data[4]
        T_collect = data[5]
        self.T_collect.append(T_collect)
        self.T_send.append(T_send)
        self.Theta1.append(Theta1)
        self.Theta2.append(Theta2)
        self.Angle_velocity1.append(dtheta1)
        self.Angle_velocity2.append(dtheta2)

    def save_to_file(self):
        np.save('data\Theta1.npy', np.array(self.Theta1))
        np.save('data\Theta2.npy', np.array(self.Theta2))
        np.save('data\Send_Torque.npy', np.array(self.T_send))
        np.save('data\Collect_Torque.npy', np.array(self.T_collect))
        np.save('data\Angle_velocity1.npy', np.array(self.Angle_velocity1))
        np.save('data\Angle_velocity2.npy', np.array(self.Angle_velocity2))
        print('数据保存成功')

    def load(self):
        Angle1 = np.load('data\Theta1.npy')
        Angle2 = np.load('data\Theta2.npy')
        Angle_velocity1 = np.load('data\Angle_velocity1.npy')
        Angle_velocity2 = np.load('data\Angle_velocity2.npy')
        send_torque = np.load('data\Send_Torque.npy')
        collet_torque = np.load('data\Collect_Torque.npy')
        return (Angle1, Angle2, Angle_velocity1, Angle_velocity2, send_torque, collet_torque)