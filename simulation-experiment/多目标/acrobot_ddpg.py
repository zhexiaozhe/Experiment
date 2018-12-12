import filter_env
import time
from numpy import pi
# gc is memory manage modle
import gc
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#增加中文功能
plt.rcParams['axes.unicode_minus']=False
import numpy as np
import gym
from ddpg import *
import scipy.io as sio

gc.enable()

dataFile = u'F:/matlab_files/多目标实验2.mat'
ENV_NAME = 'Acrobot-v1'
# ENV_NAME = 'Pendulum-v0'
test_name='多目标实验'
EPISODES = 4000
TEST = 1
#单目标
plot_reward=[]
plot_step=[]

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    for episode in range(EPISODES):
        # Train
        state = env.reset()
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state

            if done:
                break

        # Testing:
        if (episode+1) % 50 == 0 or episode==0:
            total_reward=0
            testing_step = 0
            theta1=[]
            theta2=[]
            theta1d=[]
            theta2d=[]
            Action=[]
            dtheta1=[]
            dtheta2=[]
            step=[]
            distance=[]
            for i in range(TEST):
                state = env.reset()
                test_action_smoothing=0
                for j in range(env.spec.timestep_limit):
                    env.render()
                    action = agent.action(state)
                    # critic_value=agent.critic(state,action)[0]
                    # test_action_smoothing=np.clip(0.5*action+0.5*test_action_smoothing,env.action_space.low,env.action_space.high)
                    # state,reward,done,inf = env.step(test_action_smoothing)
                    state, reward, done, inf = env.step(action)
                    step.append(0.01*j)
                    theta1.append(inf[0][0])
                    theta2.append(inf[0][1])
                    dtheta1.append(inf[0][2])
                    dtheta2.append(inf[0][3])
                    theta1d.append(inf[1])
                    theta2d.append(inf[2])
                    distance.append(inf[5])
                    Action.append(10*test_action_smoothing)

                    total_reward += reward
                    if done:
                         break
                #能量
                sio.savemat(dataFile, {'Theta1': theta1, 'Theta2': theta2, 'Theta2d': theta2d, 'Theta1d': theta1d,'Distance':distance})
                plt.figure('响应')
                plt.plot(step, theta1,'r-', label="Theta1")
                plt.plot(step, theta2,'b--', label="Theta2")
                plt.plot(step, theta1d, 'g-.',label="Theta1d")
                plt.plot(step, theta2d, 'c:',label="Theta2d")
                plt.xlabel('time/s')
                plt.ylabel('角度响应（rad）')
                plt.grid()
                plt.legend()
                plt.figure('动作')
                plt.plot(step, Action, label="Action")
                plt.grid()
                plt.legend()
                plt.figure('角速度')
                plt.plot(step, dtheta1, label='dtheta1')
                plt.plot(step, dtheta2, label='dtheta2')
                plt.grid()
                plt.legend()
                plt.figure('距离')
                plt.plot(step,distance,label="distance")
                plt.legend()
                # plt.figure('能量')
                # plt.plot(step, Ed, label="Ed")
                # plt.plot(step, E, label="E")
                # plt.legend()
                # plt.grid()
                plt.show()

                #基于距离和力矩
                # plt.figure('响应')
                # plt.plot(step, theta1, label="Theta1")
                # plt.plot(step, theta2, label="Theta2")
                # plt.plot(step, theta1d, label="Theta1")
                # plt.plot(step, theta2d, label="Theta2")
                # plt.grid()
                # plt.legend()
                # plt.figure('动作')
                # plt.plot(step, Action, label="Action")
                # plt.grid()
                # plt.legend()
                # plt.figure('角速度')
                # plt.plot(step, dtheta1, label='dtheta1')
                # plt.plot(step, dtheta2, label='dtheta2')
                # plt.grid()
                # plt.legend()
                # plt.show()

            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:',
                ave_reward,time.strftime('  %Y-%m-%d %A %X',time.localtime()))

            plot_reward.append(ave_reward)

    np.save('data\%s.npy'%test_name, np.array(plot_reward))
    plt.figure(1)
    plt.plot(plot_reward)
    plt.grid()
    plt.xlabel('step')
    plt.ylabel('average_reward')
    plt.savefig('figure\%s'%test_name)
    plt.show()

if __name__ == '__main__':
    main()