import filter_env
import time
import gc
import matplotlib.pyplot as plt
import numpy as np
import gym
import scipy.io as sio
from ddpg import *
from numpy import sin,pi
plt.rcParams['font.sans-serif']=['SimHei']#增加中文功能
plt.rcParams['axes.unicode_minus']=False
gc.enable()

ENV_NAME = 'Acrobot-v1'
# ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'MountainCarContinuous-v0'
test_name='实物实验1'
EPISODES = 2000
TEST = 1
theta1=[]
theta2=[]
theta2d=[]
theta1d=[]
energy=[]
Ed=[]
times=[]
fontsize=10
plot_reward=[]
def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    for episode in range(EPISODES):
        # Training
        state = env.reset()
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done) #if env = = cartpole need next_state flatten
            state = next_state
            if done:
                break

        # Testing:
        if (episode+1) % 50 == 0 or episode==0:
            total_reward=0
            for i in range(TEST):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    env.render()
                    if j<700:
                        action=0.7 * sin(0.01*pi * j)
                        # action=-0.8
                    else:
                        action = agent.action(state)
                    # action = agent.action(state)
                    state, reward, done, inf = env.step(action)
                    total_reward += reward
                    theta1.append(inf[0][0])
                    theta2.append(inf[0][1])
                    theta1d.append(inf[1])
                    theta2d.append(inf[2])
                    energy.append(inf[5])
                    Ed.append(inf[6])
                    times.append(0.01 * j)
                    if done:
                         break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:',
                ave_reward,time.strftime('  %Y-%m-%d %A %X',time.localtime()))
            plot_reward.append(ave_reward)
            # plt.figure('响应曲线')
            # plt.plot(times, theta1, 'r--', label=r'$\theta_1$')
            # plt.plot(times, theta2, 'b-', label=r'$\theta_2$')
            # plt.plot(times, theta1d, 'r--', label=r'$\theta_1d$')
            # plt.plot(times, theta2d, 'b-', label=r'$\theta_2d$')
            # plt.legend()
            # plt.xlabel('Time/s', fontsize=fontsize)
            # plt.ylabel('Angle/rad', fontsize=fontsize)
            # plt.xticks(fontsize=fontsize)
            # plt.yticks(fontsize=fontsize)
            # plt.figure('能量')
            # plt.plot(times,energy,'r--',label=r'energy')
            # plt.plot(times,Ed,'b-',label=r'Ed')
            # plt.legend()
            # # dataFile = u'F:/matlab_files/实物实验3.mat'
            # # sio.savemat(dataFile, {'Theta1': theta1, 'Theta2': theta2, 'Time': times})
            # plt.show()
if __name__ == '__main__':
    main()