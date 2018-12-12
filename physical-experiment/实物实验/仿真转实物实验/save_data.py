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
        self.DIS=[]
        self.Theta1d=[]
        self.Theta2d=[]
        self.Time=[]

    def recorde(self,data):
        Theta1=data[0]
        if Theta1>=3.0:
            Theta1=Theta1-2*np.pi
        Theta2=data[1]
        dtheta1=data[2]
        dtheta2=data[3]
        T_send = data[4]
        T_collect = data[5]
        dis=data[6]
        theta1d=data[7]
        theta2d=data[8]
        time=data[9]*0.01

        self.T_collect.append(T_collect)
        self.T_send.append(2.73*T_send)
        self.Theta1.append(Theta1)
        self.Theta2.append(Theta2)
        self.Angle_velocity1.append(dtheta1)
        self.Angle_velocity2.append(dtheta2)
        self.DIS.append(dis)
        self.Theta1d.append(theta1d)
        self.Theta2d.append(theta2d)
        self.Time.append(time)

    def save_to_file(self):
        np.save('data\Theta1.npy', np.array(self.Theta1))
        np.save('data\Theta2.npy', np.array(self.Theta2))
        np.save('data\Send_Torque.npy', np.array(self.T_send))
        np.save('data\Collect_Torque.npy', np.array(self.T_collect))
        np.save('data\Angle_velocity1.npy', np.array(self.Angle_velocity1))
        np.save('data\Angle_velocity2.npy', np.array(self.Angle_velocity2))
        np.save('data\distance.npy', np.array(self.DIS))
        np.save('data\Theta1d.npy', np.array(self.Theta1d))
        np.save('data\Theta2d.npy', np.array(self.Theta2d))
        np.save('data\Time.npy',np.array(self.Time))
        print('数据保存成功')

    def load(self):
        Angle1 = np.load('data\Theta1.npy')
        Angle2 = np.load('data\Theta2.npy')
        Angle_velocity1 = np.load('data\Angle_velocity1.npy')
        Angle_velocity2 = np.load('data\Angle_velocity2.npy')
        send_torque = np.load('data\Send_Torque.npy')
        collet_torque = np.load('data\Collect_Torque.npy')
        distance=np.load('data\distance.npy')
        Angle1d=np.load('data\Theta1d.npy')
        Angle2d = np.load('data\Theta2d.npy')

        return (Angle1, Angle2, Angle_velocity1, Angle_velocity2, send_torque, collet_torque,distance,Angle1d,Angle2d)