# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/4/17 16:37
'''
"""classic Acrobot task"""
from gym import core, spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
import time
import math
__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class AcrobotEnv(core.Env):

    """
    Acrobot is a 2-link pendulum with only the second joint actuated
    Intitially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    The action is either applying +1, 0 or -1 torque on the joint between
    the two pendulum links.
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondance
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    dt = .02
    #基于能量的理想模型
    # LINK_LENGTH_1 = 0.5 # [m]
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
    # g = 9.81
    #理想并且没有摩擦的模型参数
    # LINK_LENGTH_1 = 1.  # [m]
    # LINK_LENGTH_2 = 1.  # [m]
    # LINK_MASS_1 = 1.  #: [kg] mass of link 1
    # LINK_MASS_2 = 1.  #: [kg] mass of link 2
    # LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    # LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    # LINK_MOI1 = 1.  #: moments of inertia for both links
    # LINK_MOI2 = 1.
    # g = 9.81
    #有摩擦实际模型
    LINK_LENGTH_1 = 0.593  # [m]
    LINK_LENGTH_2 = 0.593  # [m]
    LINK_MASS_1 = 2.73  #: [kg] mass of link 1
    LINK_MASS_2 = 1.68  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.3  #: [m] position of the center of mass of link 1
    LINK_MOI1 = 0.25  #: moments of inertia for both links
    LINK_COM_POS_2 = 0.4  #: [m] position of the center of mass of link 2
    LINK_MOI2 = 0.116
    MU11 = 0.205
    MU12=0.184
    MU21=0.93
    MU22=1.07
    g = 9.81

    omega1=LINK_MASS_1*LINK_COM_POS_1**2+LINK_MASS_2*LINK_LENGTH_1**2+LINK_MOI1
    omega2=LINK_MASS_2*LINK_COM_POS_2**2+LINK_MOI2
    omega3=LINK_MASS_2*LINK_LENGTH_1*LINK_COM_POS_2
    omega4=LINK_MASS_1*LINK_COM_POS_1+LINK_MASS_2*LINK_LENGTH_1
    omega5=LINK_MASS_2*LINK_COM_POS_2
    theta1d = -pi / 4
    theta2d = pi / 2

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi
    max_torque = 10.
    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        self.viewer = None
        # high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2,1.0,1.0,1.0,1.0])
        high=np.array([1.0,1.0,1.0,1.0,self.MAX_VEL_1,self.MAX_VEL_2])
        low = -high
        self.observation_space = spaces.Box(low, high)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.state = None
        self._seed()
        self.radius = 0.05
        self.ns_smoothing=0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        # high = np.array([-3*pi / 4 + 0.2, -pi/2+0.2, 0.0, 0.0])
        # low = np.array([-3*pi / 4 - 0.2, -pi/2-0.2, -0.0, -0.0])
        high = np.array([-pi/2+0.1, 0.1, 0.0, 0.0])
        low = np.array([-pi/2-0.1, -0.1, -0.0, -0.0])
        # high = np.array([-7 * pi / 8 + 0.3, -pi / 4 + 0.1, 0.0, 0.0])
        # low = np.array([-7*pi / 8 - 0.1, -pi/4-0.1, -0.0, -0.0])
        self.state = self.np_random.uniform(low=low, high=high)
        # self.state=[-3*pi/4,-pi/4,0,0]
        # self.state=[-pi/2,0,0,0]
        # self.theta1d=self.np_random.uniform(low=-pi/3, high=-pi/6, size=(1,))[0]
        # self.theta2d = self.np_random.uniform(low=pi/3, high=2*pi/3, size=(1,))[0]
        #虚约束的方法的参数
        # self.a = (self.theta1d+pi/2) / self.theta2d
        # self.b = -pi/2
        # self.A = self.omega2 + self.a * self.omega1 + self.a * self.omega2
        # self.B = (1 + 2 * self.a) * self.omega3
        # self.Y0 =0
        # self.state = [self.theta1d, self.theta2d, 0, 0]
        return self._get_ob()

    def _step(self, a):

        s = self.state
        #对称虚约束的方法
        # phi=self.a*s[1]+self.b
        # Y=-(2*self.g*(self.F(s[1])-self.F(self.theta2d)))/(self.A+self.B*cos(s[1]))**2+self.Y0*((self.A+self.B*cos(self.theta2d))/(self.A+self.B*cos(s[1])))**2
        # reward1=-abs(phi-s[0])
        # reward2=-abs(Y-s[3]**2)
        # reward=10*reward1+reward2
        # p=self.state[3]*a[0]
        # self.consume_energy+=p*0.02

        # 目标能量
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI1
        I2 = self.LINK_MOI2
        g = 9.81
        Ed = m1 * g * lc1 * sin(self.theta1d) + m2 * g * (l1 * sin(self.theta1d) + lc2 * sin(self.theta1d + self.theta2d))
        torque = np.clip(a, -self.max_torque, self.max_torque)[0]
        E = self.energy(s)
        # 只考虑抓取
        D = self.distance(self.state)
        # if D < 0.05:
        #     reward = 100 - 0.1 * torque ** 2-self.state[2]**2-self.state[3]**2
        #     done = 1
        # else:
        #     reward = -10 * D - 0.1 * torque ** 2
        #     done = 0
        #基于能量的变目标

        #基于能量和目标点
        # if D<0.05:
        #     reward = 100 - 0.1 * torque ** 2 - self.state[2] ** 2 - self.state[3] ** 2
        #     done=1
        # elif E<1.5*Ed:
        reward1 = -abs(E - Ed)
        reward2 = -6 * abs(self.state[1] - 0)
        reward3 = -0.5 * abs(torque - 0)
        reward = reward1 + reward2 + reward3
        done=0
        # else:
        #     reward1 = -abs(E - Ed)
        #     reward2 = -5 * abs(self.state[1] - self.theta2d)
        #     reward = reward1 + reward2
        #     done=0

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        #只考虑抓取
        # return (self._get_ob(),reward,done,[self.state,self.theta1d,self.theta2d,D,reward])
        #只考虑起摆
        # return (self._get_ob(),reward,False,[E,self.state,Ed])
        #基于能量
        return (self._get_ob(), reward, done, [self.state,self.theta1d,self.theta2d,E,Ed,D])
        # 对称虚约束的
        # return (self._get_ob(), reward, False,[reward1,reward2,self.state,phi,Y,D])

    def F(self,x):
        F=(self.A*self.omega4+0.5*self.B*self.omega5)/self.a*sin(self.a*x+self.b)+(self.A*self.omega5+0.5*self.B*self.omega4)/(self.a+1)*sin((self.a+1)*x+self.b)\
          +0.5*self.B*self.omega4/(self.a-1)*sin((self.a-1)*x+self.b)+0.5*self.B*self.omega5/(self.a+2)*sin((self.a+2)*x+self.b)
        return F

    def distance(self,s):
        # 模型改目标的位置
        g_p1 = [self.LINK_LENGTH_1 *
                np.sin(self.theta1d), self.LINK_LENGTH_1 * np.cos(self.theta1d)]

        g_p2 = [g_p1[0] + self.LINK_LENGTH_2 * np.sin(self.theta1d + self.theta2d),
                g_p1[1] + self.LINK_LENGTH_2 * np.cos(self.theta1d + self.theta2d)]
        # 模型改末端位置
        p1 = [self.LINK_LENGTH_1 *
              np.sin(s[0]), self.LINK_LENGTH_1 * np.cos(s[0])]

        p2 = [p1[0] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.cos(s[0] + s[1])]
        # 末端与目标点之间的距离
        D = math.sqrt((p2[0] - g_p2[0]) ** 2 + (p2[1] - g_p2[1]) ** 2)
        return D

    def _get_ob(self):
        s = self.state
        return np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2],s[3]])

    #变目标
    # def _get_ob(self):
    #     s = self.state
    #     return np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3],\
    #                      sin(self.theta1d),cos(self.theta1d),sin(self.theta2d),cos(self.theta2d)])

    def _terminal(self):
        s = self.state
        return bool(s[0]>0 and s[2]==0 and s[3]==0)
        # return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI1
        I2 = self.LINK_MOI2
        mu11=self.MU11
        mu12=self.MU12
        mu21=self.MU21
        mu22=self.MU22

        g = 9.81
        a = s_augmented[-1]
        s = s_augmented[:-1]
        ########################################
        #全驱动
        # a=s_augmented[-1]
        # s = s_augmented[:-1]
        # tau1=a
        # tau2=-5
        #######################################
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        #模型改1(在模型改的基础上增加摩擦)
        sgn=lambda x:1 if x>0 else -1 if x<0 else 0
        d11=self.omega1+self.omega2+2*self.omega3*cos(theta2)
        d12=self.omega2+self.omega3*cos(theta2)
        d21=d12
        d22=self.omega2
        h1=-self.omega3*dtheta2*sin(theta2)*(2*dtheta1+dtheta2)+mu11*dtheta1+mu12*sgn(dtheta1)
        h2=self.omega3*dtheta1**2*sin(theta2)+mu21*dtheta2+mu22*sgn(dtheta2)
        phi2=self.omega5*g*cos(theta1+theta2)
        phi1=self.omega4*g*cos(theta1)+phi2
        ddtheta1 = (d12 * a - d12 * (h2 + phi2) + d22 * (h1 + phi1)) / (d12 * d21 - d22 * d11)
        ddtheta2 = (d21 * (h1 + phi1) - d11 * (h2 + phi2) + d11 * a) / (d11 * d22 - d12 * d21)

        #全驱动
        # ddtheta1 = (d12 * tau2 - d22 *a  - d12 * (h2 + phi2) + d22 * (h1 + phi1)) / (d12 * d21 - d22 * d11)
        # ddtheta2 = (d21 * (h1 + phi1) - d11 * (h2 + phi2) + d11 * tau2 - d21 *a ) / (d11 * d22 - d12 * d21)
        #模型改（参考角度不同）
        # sgn = lambda x: 1 if x > 1 else -1 if x < -1 else 0
        # d1 = m1 * lc1 ** 2 + m2 * \
        #                      (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        # d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2))
        # phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 )
        # phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
        #        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) \
        #        + (m1 * lc1 + m2 * l1) * g * np.cos(theta1) + phi2
        # if self.book_or_nips == "nips":
        #     # the following line is consistent with the description in the
        #     # paper
        #     ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
        #                (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        # else:
        #     # the following line is consistent with the java implementation and the
        #     # book
        #     ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
        #                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        # ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        # 原模型
        # d1 = m1 * lc1 ** 2 + m2 * \
        #     (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        # d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        # phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        # phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
        #        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)  \
        #     + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        # if self.book_or_nips == "nips":
        #     # the following line is consistent with the description in the
        #     # paper
        #     ddtheta2 = (a + d2 / d1 * phi1 - phi2) / \
        #         (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        # else:
        #     # the following line is consistent with the java implementation and the
        #     # book
        #     ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
        #         / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        # ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2*self.LINK_LENGTH_1,2.2*self.LINK_LENGTH_1,-2.2*self.LINK_LENGTH_1,2.2*self.LINK_LENGTH_1)
            #self.viewer.set_bounds(-1.5, 1.5, -1.5, 1.5)#模型改采用这个比例
        if s is None: return None

        # p1 = [-self.LINK_LENGTH_1 *
        #       np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]
        #
        # p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
        #       p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]
        #模型改
        p1 = [self.LINK_LENGTH_1 *
              np.sin(s[0]), self.LINK_LENGTH_1 * np.cos(s[0])]

        p2 = [p1[0] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.cos(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        # thetas = [s[0]-np.pi/2, s[0]+s[1]-np.pi/2]
        thetas = [s[0], s[0] + s[1]] #模型改
        # goal dot
        g_p1 = [self.LINK_LENGTH_1 *
              np.sin(self.theta1d), self.LINK_LENGTH_1 * np.cos(self.theta1d)]

        g_p2 = [g_p1[0] + self.LINK_LENGTH_2 * np.sin(self.theta1d + self.theta2d),
              g_p1[1] + self.LINK_LENGTH_2 * np.cos(self.theta1d + self.theta2d)]

        circ_goal = self.viewer.draw_circle(self.radius)
        circ_goal.set_color(.8, .4, 0)
        jtransform_goal = rendering.Transform(rotation=.0, translation=(g_p2[1], g_p2[0]))
        circ_goal.add_attr(jtransform_goal)

        # self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th) in zip(xys, thetas):
            l,r,t,b = 0, self.LINK_LENGTH_1, .05, -.05
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.05)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def energy(self, s):
        # energy function
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI1
        I2 = self.LINK_MOI2
        g = 9.81

        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        # T1=1/2*I1*dtheta1**2+1/2*m1*(dtheta1*lc1)**2
        # V1=-m1*g*lc1*cos(theta1)
        # T2=1/2*m2*((dtheta1*l1)**2+((dtheta1+dtheta2)*lc2)**2+2*dtheta1*(dtheta1+dtheta2)*lc2*l1*cos(theta2))+1/2*I2*(dtheta1+dtheta2)**2
        # V2=-m2*g*(l1*cos(theta1)+lc2*cos(theta1+theta2))
        # E=V1+V2+T1+T2
        #模型改
        # T=1/2*I1*dtheta1**2+1/2*I2*(dtheta1+dtheta2)**2+1/2*m1*lc1**2*dtheta1**2\
        #   +1/2*m2*(l1**2*dtheta1**2+lc2**2*(dtheta1+dtheta2)**2+2*l1*lc2*dtheta1*(dtheta1+dtheta2)*cos(theta2))
        # V=m1*g*lc1*sin(theta1)+m2*g*(l1*sin(theta1)+lc2*sin(theta1+theta2))
        # E=T+V
        #模型改2
        T1 = 1 / 2 * I1 * dtheta1 ** 2 + 1 / 2 * m1 * (dtheta1 * lc1) ** 2
        V1 = m1 * g * lc1 * sin(theta1)
        T2 = 1 / 2 * m2 * ((dtheta1 * l1) ** 2 + ((dtheta1 + dtheta2) * lc2) ** 2 + 2 * dtheta1 * (dtheta1 + dtheta2)
                           * lc2 * l1 * cos(theta2)) + 1 / 2 * I2 * (dtheta1 + dtheta2) ** 2
        V2 = m2 * g * (l1 * sin(theta1) + lc2 * sin(theta1 + theta2))
        E = V1 + V2 + T1 + T2
        return E

def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0
    i = 0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
if __name__=='__main__':
    env=AcrobotEnv()
    env.reset()
    env.render()
    time.sleep(100)