# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/23 9:30
'''
import numpy as np
import gym
import matplotlib.pyplot as plt

from numpy import sin,cos,pi

# 目标点
theta1d = -pi / 4
theta2d = pi / 2

class CONTROL(object):
    #系统参数
    # LINK_LENGTH_1 = 0.5  # [m]
    # LINK_LENGTH_2 = 0.5  # [m]
    # LINK_MASS_1 = 3.499  #: [kg] mass of link 1
    # LINK_MASS_2 = 1.232  #: [kg] mass of link 2
    # LINK_COM_POS_1 = 0.141  #: [m] position of the center of mass of link 1
    # LINK_COM_POS_2 = 0.333  #: [m] position of the center of mass of link 2
    # LINK_MOI1 = 0.09  #: moments of inertia for both links
    # LINK_MOI2 = 0.033
    # MU11 = 0.
    # MU12 = 0.
    # MU21 = 0.
    # MU22 = 0.

    LINK_LENGTH_1 = 0.593
    LINK_LENGTH_2 = 0.593
    LINK_MASS_1 = 2.73
    LINK_MASS_2 = 1.68
    LINK_COM_POS_1 = 0.4
    LINK_MOI1 = 0.25
    LINK_COM_POS_2 = 0.377
    LINK_MOI2 = 0.116
    MU11 = 0.205
    MU12 = 0.184
    MU21 = 0.93
    MU22 = 1.07

    omega1 = LINK_MASS_1 * LINK_COM_POS_1 ** 2 + LINK_MASS_2 * LINK_LENGTH_1 ** 2 + LINK_MOI1
    omega2 = LINK_MASS_2 * LINK_COM_POS_2 ** 2 + LINK_MOI2
    omega3 = LINK_MASS_2 * LINK_LENGTH_1 * LINK_COM_POS_2
    omega4 = LINK_MASS_1 * LINK_COM_POS_1 + LINK_MASS_2 * LINK_LENGTH_1
    omega5 = LINK_MASS_2 * LINK_COM_POS_2
    g = 9.81

    def __init__(self):
        self.name="control"
        # self.KE=0.2
        # self.KD=15
        # self.KP=44
        # self.KV=45
        self.KE =0.02
        self.KD=15
        self.KP=50
        self.KV=45

    def energy(self,state):

        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI1
        I2 = self.LINK_MOI2
        g = 9.81

        theta1 = state[0]
        theta2 = state[1]
        dtheta1 = state[2]
        dtheta2 = state[3]

        T1 = 1 / 2 * I1 * dtheta1 ** 2 + 1 / 2 * m1 * (dtheta1 * lc1) ** 2
        V1 = m1 * g * lc1 * sin(theta1)
        T2 = 1 / 2 * m2 * ((dtheta1 * l1) ** 2 + ((dtheta1 + dtheta2) * lc2) ** 2 + 2 * dtheta1 *
                           (dtheta1 + dtheta2) * lc2 * l1 * cos(theta2)) + 1 / 2 * I2 * (dtheta1 + dtheta2) ** 2
        V2 = m2 * g * (l1 * sin(theta1) + lc2 * sin(theta1 + theta2))
        E = V1 + V2 + T1 + T2
        return E

    def Ed(self):

        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        g = 9.81
        Ed = m1 * g * lc1 * sin(theta1d) + m2 * g * (l1
                    * sin(theta1d) + lc2 * sin(theta1d + theta2d))
        return Ed

    def T(self,state):

        theta1 = state[0]
        theta2 = state[1]
        dtheta1 = state[2]
        dtheta2 = state[3]

        sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0
        d11 = self.omega1 + self.omega2 + 2 * self.omega3 * cos(theta2)
        d12 = self.omega2 + self.omega3 * cos(theta2)
        d21 = d12
        d22 = self.omega2
        h1 = -self.omega3 * dtheta2 * sin(theta2) * (2 * dtheta1 + dtheta2) + self.MU11 * dtheta1 + self.MU12 * sgn(
            dtheta1)
        h2 = self.omega3 * dtheta1 ** 2 * sin(theta2) + self.MU21 * dtheta2 + self.MU22 * sgn(dtheta2)
        phi2 = self.omega5 * self.g * cos(theta1 + theta2)
        phi1 = self.omega4 * self.g * cos(theta1) + phi2

        delta=d11*d22-d12**2
        eE=self.energy(state)
        edq2=theta2-theta2d

        tau=(-self.KV*dtheta2-self.KP*edq2-self.KD/delta*(d21*(h1+phi1)-d11*(h2+phi2)))/(self.KE*eE+self.KD*d11/delta)
        return tau

if __name__=='__main__':
    env=gym.make('Acrobot-v1')
    state=env.reset()
    control=CONTROL()
    Theta1 = []
    Theta2 = []
    Theta1d = []
    Theta2d = []
    Theta1_velocity = []
    Theta2_velocity = []
    Action = []

    for step in range(20000):
        # env.render()
        T=[np.clip(control.T(state),-15,15)]
        obs,r,done,inf=env.step(T)
        Theta1.append(inf[1][0])
        Theta2.append(inf[1][1])
        Theta1_velocity.append(inf[1][2])
        Theta2_velocity.append(inf[1][3])
        Theta1d.append(theta1d)
        Theta2d.append(theta2d)
        Action.append(T)
        state=inf[1]
        if done:
            break
    plt.figure('响应')
    plt.plot(Theta1,'r-',label='Theta1')
    plt.plot(Theta1d,'g--',label='Theta1d')
    plt.plot(Theta2, 'b-', label='Theta2')
    plt.plot(Theta2d,'y--',label='Theta2d')
    plt.legend()
    plt.figure('动作')
    plt.plot(Action,'r-',label='Action')
    plt.legend()
    plt.figure('角速度')
    plt.plot(Theta1_velocity,'r-',label='Theta1_volocity')
    plt.plot(Theta2_velocity, 'b-', label='Theta2_volocity')
    plt.legend()
    plt.show()
