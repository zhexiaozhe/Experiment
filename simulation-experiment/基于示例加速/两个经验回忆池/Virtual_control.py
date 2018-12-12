# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@file: 虚约束控制.py
@time: 2017/11/18 14:46
'''

import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle

from numpy import pi,sin,cos
from math import atan
from collections import deque

#虚约束控制函数
class CONTROL:
    #系统参数
    LINK_LENGTH_1 = 0.593
    LINK_LENGTH_2 = 0.593
    LINK_MASS_1 = 2.78
    LINK_MASS_2 = 1.73
    LINK_COM_POS_1 = 0.3
    LINK_MOI1 = 0.25
    LINK_COM_POS_2 = 0.4
    LINK_MOI2 = 0.116
    MU11 = 0.205
    MU12 = 0.184
    MU21 = 0.93
    MU22 = 1.07
    g = 9.81
    omega1 = LINK_MASS_1 * LINK_COM_POS_1 ** 2 + LINK_MASS_2 * LINK_LENGTH_1 ** 2 + LINK_MOI1
    omega2 = LINK_MASS_2 * LINK_COM_POS_2 ** 2 + LINK_MOI2
    omega3 = LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_POS_2
    omega4 = LINK_MASS_1 * LINK_COM_POS_1 + LINK_MASS_2 * LINK_LENGTH_1
    omega5 = LINK_MASS_2 * LINK_COM_POS_2
    #目标点
    theta1d = -pi / 4
    theta2d = pi / 2
    #虚约束参数
    a = (theta1d + pi / 2) / theta2d
    b = -pi / 2
    def __init__(self):
        self.name="control"

    def F(self,x):
        F=(self.A*self.omega4+0.5*self.B*self.omega5)/self.a*sin(self.a*x+self.b)+(self.A*self.omega5+0.5*self.B*self.omega4)/(self.a+1)*sin((self.a+1)*x+self.b)\
          +0.5*self.B*self.omega4/(self.a-1)*sin((self.a-1)*x+self.b)+0.5*self.B*self.omega5/(self.a+2)*sin((self.a+2)*x+self.b)
        return F

    def U(self,s):
        self.s=s
        self.A = self.omega2 + self.a * self.omega1 + self.a * self.omega2
        self.B = (1 + 2 * self.a) * self.omega3
        self.Y0 =0
        Y = -(2 * self.g * (self.F(self.s[1]) - self.F(self.theta2d))) / (self.A + self.B * cos(self.s[1])) ** 2 + \
            self.Y0 * ((self.A + self.B * cos(self.theta2d)) / (self.A + self.B * cos(self.s[1]))) ** 2
        U=self.s[3]**2-Y
        return U

    def Torque(self,state):
        theta1=state[0]
        theta2=state[1]
        dtheta1=state[2]
        dtheta2=state[3]
        ############################################
        #控制器系数
        k1=20
        k2=80
        k3=55
        #############################################
        sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0
        d11 = self.omega1 + self.omega2 + 2 * self.omega3 * cos(theta2)
        d12 = self.omega2 + self.omega3 * cos(theta2)
        d21 = d12
        d22 = self.omega2
        h1 = -self.omega3 * dtheta2 * sin(theta2) * (2 * dtheta1 + dtheta2) +self.MU11 * dtheta1 + self.MU12 * sgn(dtheta1)
        h2 = self.omega3 * dtheta1 ** 2 * sin(theta2) + self.MU21 * dtheta2 + self.MU22 * sgn(dtheta2)
        phi2 = self.omega5 * self.g * cos(theta1 + theta2)
        phi1 = self.omega4 * self.g * cos(theta1) + phi2
        U=self.U(state)
        h=theta1-self.b-self.a*theta2
        dh=dtheta1-self.a*dtheta2
        B1=d11+d12
        B2=-self.omega3*sin(theta2)*(2*self.a+1)
        B3=self.omega4*self.g*cos(self.a*theta2+self.b)+self.omega5*self.g*cos(self.a*theta2+self.b+theta2)
        fh=B3-self.omega4*self.g*cos(h+self.a+self.b)+self.omega5*self.g*cos(h+self.a*theta2+self.b+theta2)
        fdh=2*self.omega3*dtheta2*sin(theta2)*dh
        v=(k3*atan(U*dtheta2/B1)-B2*U+fh+fdh)/d11#虚约束项
        f1=-(d22*(h1+phi1)-d12*(h2+phi2))/(d11*d22-d12*d21)
        g1=-d12/(d11*d22-d12*d21)
        f2=(d21*(h1+phi1)-d11*(h2+phi2))/(d11*d22-d12*d21)
        g2=d11/(d11*d22-d12*d21)
        ddh=v-k1*dh-k2*h
        T=(ddh-f1+self.a*f2)/(g1-self.a*g2)
        return T

if __name__ == '__main__':
    env=gym.make('Acrobot-v1')
    control=CONTROL()
    #采集数据
    buffer = deque()
    normal = [1, 1, 1, 1, 4 * np.pi, 9 * np.pi]
    #绘图变量
    Theta1=[]
    Theta2=[]
    Theta1d=[]
    Theta2d=[]
    Theta1_velocity=[]
    Theta2_velocity=[]
    Action=[]
    Action_smoothing=[]
    Tau_theta1=[]
    for _ in range(500):
        state = env.reset()
        print(_)
        for step in range(1000):
            # env.render()
            if step==0:
                action=[0]
            else:
                action=[np.clip(control.Torque(inf[0]),-10,10)]
            next_state, r, done, inf = env.step(action)
            action=np.array(action)/10
            buffer.append((state / normal, action, r, next_state / normal, done))
            # Theta1.append(inf[0][0])
            # Theta2.append(inf[0][1])
            # Theta1_velocity.append(inf[0][2])
            # Theta2_velocity.append(inf[0][3])
            # Action.append(action)
            if done:
                break
            state = next_state
        # plt.figure(1)
        # plt.plot(Theta1,'r-',label='Theta1')
        # plt.plot(Theta1d,'g--',label='Theta1d')
        # plt.plot(Theta2, 'b-', label='Theta2')
        # plt.plot(Theta2d,'y--',label='Theta2d')
        # plt.legend()
        # plt.figure(2)
        # plt.plot(Action,'r-',label='Action')
        # plt.show()
    pickle.dump(buffer, open('data\object.pickle', 'wb'))
        # Tau_theta1.append(a_smoothing[0]*inf[0][2])
        # Theta1.append(inf[0][0])
        # Theta2.append(inf[0][1])
        # Theta1_velocity.append(inf[0][2])
        # Theta2_velocity.append(inf[0][3])
        # Theta1d.append(-pi/4)
        # Theta2d.append(pi/2)
        # Action.append(a)
        # Action_smoothing.append(a_smoothing)

    ###################################
    # plt.figure(1)
    # plt.plot(Theta1,'r-',label='Theta1')
    # plt.plot(Theta1d,'g--',label='Theta1d')
    # plt.plot(Theta2, 'b-', label='Theta2')
    # plt.plot(Theta2d,'y--',label='Theta2d')
    # plt.legend()
    # plt.figure(2)
    # plt.plot(Action,'r-',label='Action')
    # plt.plot(Action_smoothing,'b-',label='Action_smoothing')
    # plt.legend()
    # plt.figure(3)
    # plt.plot(Theta1_velocity,'r-',label='Theta1_volocity')
    # plt.plot(Theta2_velocity, 'b-', label='Theta2_volocity')
    # plt.legend()
    # plt.figure(4)
    # plt.plot(Tau_theta1,label='Tau*Theta1')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # #################################








