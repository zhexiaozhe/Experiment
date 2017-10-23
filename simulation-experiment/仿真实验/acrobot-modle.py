
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi
from scipy.integrate import odeint
import time
import math
import matplotlib.pyplot as plt

def dsdt(s_augmented, t):
    # parameter
    m1 = 2.73
    m2 = 1.68
    l1 = 0.593
    l2=0.593
    lc1 = 0.2792
    I1 = 0.1709
    lc2 = 0.4035
    I2 = 0.0701
    g = 9.81
    mu11 = 0.0514
    mu12 = 0.3987
    mu21 = 0.2176
    mu22 = 0.5948
    a = s_augmented[-1]
    s = s_augmented[:-1]
    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]

    # 猴子的动力学模型
    sgn = lambda x: 1 if x > 1 else -1 if x < -1 else 0
    omega1 = I1 + m1 * lc1 ** 2 + m2 * l1 ** 2
    omega2 = I2 + m2 * lc2 ** 2
    omega3 = m2 * l1 * lc2
    omega4 = m1 * lc1 + m2 * l1
    omega5 = m2 * lc2
    d11 = omega1 + omega2 + 2 * omega3 * cos(theta2)
    d22 = omega2
    d12 = d21 = omega2 + omega3 * cos(theta2)
    phi1 = omega4 * g * cos(theta1) + omega5 * g * cos(theta1 + theta2)
    phi2 = omega5 * g * cos(theta1 + theta2)
    h1 = -omega3 * dtheta2 * (2 * dtheta1 + dtheta2) * sin(theta2)+mu11*dtheta1+mu12*sgn(dtheta1)
    h2 = omega3 * dtheta1 ** 2 * sin(theta2)+mu21*dtheta2+mu22*sgn(dtheta2)
    ddtheta2 = (d11 * a + d21 * (h1 + phi1) - d11 * (h2 + phi2)) / (d11 * d22 - d12 *d21 )
    ddtheta1 = (d12 * a + d22 * (h1 + phi1) - d12 * (h2 + phi2)) / (d12 * d21 - d11 * d22)

    return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.)


def energy(s):
    # energy function
    m1 = 2.73
    m2 = 1.68
    l1 = 0.593
    l2 = 0.593
    lc1 = 0.2792
    lc2 = 0.4035
    I1 = 0.1709
    I2 = 0.0701
    g = 9.81

    theta1 = s[0]
    theta2 = s[1]
    dtheta1 = s[2]
    dtheta2 = s[3]

    T1 = 1 / 2 * I1 * dtheta1 ** 2 + 1 / 2 * m1 * (dtheta1 * lc1) ** 2
    V1 = m1 * g * lc1 * sin(theta1)
    T2 = 1 / 2 * m2 * ((dtheta1 * l1) ** 2 + ((dtheta1 + dtheta2) * lc2) ** 2 + 2 * dtheta1 * (dtheta1 + dtheta2) * lc2 * l1 * cos(theta2)) + 1 / 2 * I2 * (dtheta1 + dtheta2) ** 2
    V2 = m2 * g * (l1 * sin(theta1) + lc2 * sin(theta1 + theta2))
    E = V1 + V2 + T1 + T2
    return E

def wrap(x, m, M):
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def rk4(derivs, y0, t, *args, **kwargs):

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

if __name__ == '__main__':

    dt = 0.01
    state1 = [0, 0, 0, 0]
    E=[]
    q=[]
    T=[]
    for step in range(1000):
        # a=2*sin(step*0.01)
        # T.append(a)
        a=0
        e1=energy(state1)
        s_augmented1 = np.append(state1, a)
        # t1 = np.arange(0,0.005,dt)
        # ns_continuous1 = odeint(dsdt, s_augmented1, t1)
        ns_continuous1 = rk4(dsdt, s_augmented1, [0, dt])
        ns1 = ns_continuous1[-1]
        ns1 = ns1[:4]
        # ns1[0] = wrap(ns1[0], -pi, pi)
        # ns1[1] = wrap(ns1[1], -pi, pi)
        q.append([ns1[0],ns1[1]])
        state1 = ns1
        print(state1)
        E.append(e1)

        # print(e1)
    plt.plot(q)
    plt.plot(E)
    plt.figure(2)
    plt.plot(T)
    plt.show()



