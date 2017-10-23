import matplotlib.pyplot as plt
import numpy as np
from numpy import sin,cos,pi
import math

sac_x=[]
sac_y=[]
LINK_LENGTH_1=1
LINK_LENGTH_2=1
theta1=pi/6
while theta1<=pi/3:
    theta2=pi/3
    while theta2<2*pi/3:
        p1 = [-LINK_LENGTH_1 *np.cos(theta1), LINK_LENGTH_1 * np.sin(theta1)]

        p2 = [p1[0] - LINK_LENGTH_2 * np.cos(theta1 + theta2),p1[1] + LINK_LENGTH_2 * np.sin(theta1 + theta2)]
        theta2+=0.01
        sac_x.append(p2[1])
        sac_y.append(p2[0])
    theta1+=0.01

print(sac_x)
print(sac_y)
plt.scatter(0,0,c='r',s=100)
plt.scatter(sac_x,sac_y)
plt.grid()
plt.show()