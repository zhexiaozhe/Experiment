# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2017/12/23 20:07
'''
import numpy as np

class SAVE_DATA(object):
    def __init__(self):
        self.T_collect = []  # 力矩采集
        self.T_send = []
        self.Theta1 = []  # 角度1采集
        self.Theta2 = []
        self.Angle_velocity1 = []  # 角速度采集
        self.Angle_velocity2 = []
        self.Time = []

    def recorde(self,data):
        angle1=data[0]
        angle2=data[1]
        angle_velocity1=data[2]
        angle_velocity2=data[3]
        value = data[4]
        torque=data[5]
        time=data[6]
        self.T_send.append(2.73 * value)
        self.T_collect.append(torque)
        self.Theta1.append(angle1)
        self.Theta2.append(angle2)
        self.Angle_velocity1.append(angle_velocity1)
        self.Angle_velocity2.append(angle_velocity2)
        self.Time.append(time)

    def save_to_file(self):
        np.save('data\Theta1.npy', np.array(self.Theta1))
        np.save('data\Theta2.npy', np.array(self.Theta2))
        np.save('data\Send_Torque.npy', np.array(self.T_send))
        np.save('data\Collect_Torque.npy',np.array(self.T_collect))
        np.save('data\Time.npy', np.array(self.Time))
        np.save('data\Angle_velocity1.npy', np.array(self.Angle_velocity1))
        np.save('data\Angle_velocity2.npy', np.array(self.Angle_velocity2))
        print('数据保存成功')

    def load(self):
        Angle1=np.load('data\Theta1.npy')
        Angle2=np.load('data\Theta2.npy')
        Angle_velocity1=np.load('data\Angle_velocity1.npy')
        Angle_velocity2=np.load('data\Angle_velocity2.npy')
        send_torque=np.load('data\Send_Torque.npy')
        collet_torque=np.load('data\Collect_Torque.npy')
        return (Angle1,Angle2,Angle_velocity1,Angle_velocity2,send_torque,collet_torque)