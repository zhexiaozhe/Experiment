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

gc.enable()

ENV_NAME = 'Acrobot-v1'
test_name='中期实验4'
EPISODES = 2000
TEST = 1
#单目标
plot_reward=[]
plot_step=[]

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    for episode in range(EPISODES):
        # Training
        state = env.reset()
        for step in range(env.spec.timestep_limit): #Pendulum-v0:env.spec.timestep_limit=200
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done) #if env = = cartpole need next_state flatten
            # print((state,action,reward,next_state,done))
            state = next_state

            if done:
                break

        # Testing:
        if (episode+1) % 10 == 0 or episode==0:
            total_reward=0
            for i in range(TEST):
                state = env.reset()
                theta2=[]
                theta1=[]
                theta2d=[]
                theta1d=[]
                dtheta1=[]
                dtheta2 = []
                E=[]
                Ed=[]
                step=[]
                Action=[]
                distance=[]
                for j in range(env.spec.timestep_limit):
                    env.render()
                    action = agent.action(state)
                    # print(action)
                    next_state, reward, done, inf = env.step(action)
                    #能量测试
                    # step.append(j*0.01)
                    # theta1.append(inf[0][0])
                    # theta2.append(inf[0][1])
                    # dtheta1.append(inf[0][2])
                    # dtheta2.append(inf[0][3])
                    # Action.append(action)
                    # theta2d.append(inf[1])
                    # theta1d.append(inf[2])
                    # E.append(inf[3])
                    # Ed.append(inf[4])
                    # distance.append(inf[5])
                    total_reward += reward
                    if done:
                         break
                #能量
                # plt.figure('响应')
                # plt.plot(step, theta1,'r-', label="Theta1")
                # plt.plot(step, theta2,'b--', label="Theta2")
                # plt.plot(step, theta1d, 'g-.',label="Theta1d")
                # plt.plot(step, theta2d, 'c:',label="Theta2d")
                # plt.xlabel('Time/s')
                # plt.ylabel('角度（rad）')
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
                # plt.figure('距离')
                # plt.plot(step,distance,label="distance")
                # plt.xlabel('Time/s')
                # plt.ylabel('距离（m）')
                # plt.legend()
                # plt.figure('能量')
                # plt.plot(step, Ed, label="Ed")
                # plt.plot(step, E, label="E")
                # plt.legend()
                # plt.grid()
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