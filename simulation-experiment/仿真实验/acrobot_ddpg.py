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
# ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'MountainCarContinuous-v0'
test_name='示教实验9'
EPISODES = 2000
TEST = 1
#单目标
plot_reward=[]
plot_step=[]
def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    # plot_reward = np.zeros((201, 5))
    # for ex in range(5):
    #     te = 0
    for episode in range(EPISODES):
        # Train
        state = env.reset()
        for step in range(env.spec.timestep_limit): #Pendulum-v0:env.spec.timestep_limit=200
            action = agent.noise_action(state)
            # action_smoothing=np.clip(0.1*action+action_smoothing,env.action_space.low,env.action_space.high)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done) #if env = = cartpole need next_state flatten
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
                Y=[]
                dtheta1=[]
                dtheta2 = []
                E=[]
                Ed=[]
                step=[]
                R=[]
                action_figure=[]
                reward_error=[]
                distance=[]
                PHI=[]
                R1=[]
                R2=[]
                R3=[]
                R4=[]
                Tau_dtheta1=[]
                Tau_dtheta2 = []
                Action=[]
                Action_smoothing=[]
                Q_value=[]
                energy=[]
                test_action_smoothing=0
                for j in range(env.spec.timestep_limit):
                    # env.render()
                    action = agent.action(state)
                    # critic_value=agent.critic(state,action)[0]
                    # test_action_smoothing=np.clip(0.5*action+0.5*test_action_smoothing,env.action_space.low,env.action_space.high)
                    # state,reward,done,inf = env.step(test_action_smoothing)
                    next_state, reward, done, inf = env.step(action)

                    #虚约束测试
                    # Action.append(10*action)
                    # PHI.append([inf[3],inf[2][0],inf[2][1]])
                    # D.append(inf[5])
                    # Y.append([inf[4],inf[2][3]**2,inf[2][1]])
                    # theta1.append(inf[2][0])
                    # theta2.append(inf[2][1])
                    # theta1d.append(-pi/4)
                    # theta2d.append(pi/2)
                    # step.append(j)

                    #能量测试
                    # step.append(j*0.01)
                    # theta1.append(inf[0][0])
                    # theta2.append(inf[0][1])
                    # dtheta1.append(inf[0][2])
                    # dtheta2.append(inf[0][3])
                    # Action.append(action)
                    # # Action_smoothing.append(test_action_smoothing)
                    # theta2d.append(inf[1])
                    # theta1d.append(inf[2])
                    # E.append(inf[3])
                    # Ed.append(inf[4])
                    # distance.append(inf[5])

                    # Tau_dtheta1.append(inf[1][2]*action)
                    # Tau_dtheta2.append(inf[1][3]*action)
                    # Q_value.append(critic_value)
                    # R.append(inf[5])
                    # R1.append(inf[6])
                    # R2.append(inf[7])
                    # R3.append(inf[8])
                    # R4.append(inf[9])
                    # energy.append(inf[9])
                    #测试acrobot
                    #单目标
                    # energy.append([inf[0]])
                    # theta.append([inf[1][0], inf[1][1], inf[3]])
                    # Ed.append(inf[2])
                    # step.append(j)
                    # action_figure.append(20 * action)
                    # distance.append(inf[4])
                    #多目标
                    # energy.append([inf[1]])
                    # theta.append([inf[2][0],inf[2][1],])
                    # Ed.append(inf[3])

                    total_reward += reward
                    if done:
                         break
                #保存数据
                # np.save(r'F:\python_files\Experiment\physical-experiment\实物实验\仿真转实物实验\open_control\data\torque1.npy',np.array(Action))

                # np.save('data\mid\Theta1',np.array(theta1))
                # np.save('data\mid\Theta2',np.array(theta2))
                # np.save('data\mid\Theta1d',np.array(theta1d))
                # np.save('data\mid\Theta2d', np.array(theta2d))
                # np.save('data\mid\step',np.array(step))
                # np.save('data\mid\distance',np.array(distance))
                # # #能量
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
                # # plt.plot(step,Action_smoothing,label="Smoth_Action")
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

                #基于距离和力矩
                # plt.figure('响应')
                # plt.plot(step, theta1, label="Theta1")
                # plt.plot(step, theta2, label="Theta2")
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

                # plt.figure('奖励')
                # plt.title(u'奖励')
                # plt.plot(R1, label='R1')
                # plt.plot(R2, Label='R2')
                # # plt.plot(R3,Label='R3')
                # plt.plot(R4, label='R7')
                # plt.plot(R, Label='R')
                # plt.legend()
                # plt.grid()
                # plt.figure('值函数')
                # plt.plot(Q_value, label='Q')
                # plt.legend()
                # plt.grid()

                #虚约束与能量结合
                # np.save('data/T.npy',np.array(Action_smoothing))
                # plt.figure(1)
                # plt.plot(step,PHI,label="Phi")
                # plt.plot(step, theta1, label="Theta1")
                # plt.plot(step, theta2, label="Theta2")
                # plt.plot(step,theta1d,label="Theta1d")
                # plt.plot(step,theta2d,label="Theta2d")
                # plt.legend(prop={'size': 15})
                # plt.figure(2)
                # plt.plot(step,energy,label="energy")
                # plt.plot(step,Ed,label="Ed")
                # plt.plot(step, theta1, label="theta1")
                # plt.legend(prop={'size': 6})
                # plt.figure(3)
                # plt.plot(Action,label="action")
                # plt.plot(Action_smoothing,label="action_smoothing")
                # # plt.scatter(step,Action,label="action")
                # plt.legend(prop={'size': 6})
                # plt.figure(4)
                # plt.plot(step, R1, label="reward1")
                # plt.plot(step, R2, label="reward2")
                # plt.legend(prop={'size': 6})
                # plt.figure(5)
                # plt.plot(step,distance,label="distance")
                # plt.figure(6)
                # plt.plot(step, dtheta1, label="dtheta1")
                # plt.plot(step, dtheta2, label="dtheta2")
                # plt.legend(prop={'size': 15})
                # plt.show()

                #虚约束
                # plt.figure(2)
                # plt.plot(step,Y)
                # plt.figure(3)
                # plt.title('Ation Figure')
                # plt.xlabel('Step')
                # plt.ylabel('Action/N·m')
                # plt.ylim(-10,10)
                # plt.plot(step,Action,'r-',label='Action')
                # plt.grid()
                # plt.legend()
                # plt.figure(4)
                # plt.title('Distance Figure')
                # plt.xlabel('Step')
                # plt.ylabel('Distance/m')
                # plt.plot(step,D,'b-',label='distance')
                # plt.grid()
                # plt.legend()

                # plt.figure(5)
                # plt.title('Angle Figure')
                # plt.xlabel('step')
                # plt.ylabel('Angle/rad')
                # plt.plot(step,theta1,'r-',label='Theta1')
                # plt.plot(step,theta2,'b-',label='Theta2')
                # plt.plot(step,theta1d,'c--',label='Theta1d')
                # plt.plot(step,theta2d,'g--',label='Theta2d')
                # plt.grid()
                # plt.legend()
                # plt.show()

                #    plt.subplot(211)
                #    plt.scatter(j, action)
                #    plt.subplot(212)
                #    plt.scatter(j, critic_value)
                #    plt.pause(0.1)
                # plt.close(1)
                # acrobot
                # plt.figure(1)
                # plt.subplot(211)
                # plt.plot(step,energy,label="energy")
                # plt.plot(step,Ed,'r--',label="Ed")
                # plt.xlabel('step')
                # plt.ylim(-20, 20)
                # plt.ylabel('energy')
                # plt.subplot(212)
                # plt.plot(theta2)
                # plt.ylabel('angle/rad')
                # plt.xlabel('step')
                # plt.figure(2)
                # plt.plot(step,Action,label="action")
                # plt.ylabel('action(N·m)')
                # plt.xlabel('step')
                # plt.figure(3)
                # plt.plot(step,distance)
                # plt.ylabel('Distance(m)')
                # plt.xlabel('step')
                # plt.show()

                #显示误差等
                # plt.plot(reward_error)
                # plt.show()

            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:',
                ave_reward,time.strftime('  %Y-%m-%d %A %X',time.localtime()))
            #单目标
            # plot_reward[te][ex]=ave_reward
            # te+=1
            plot_reward.append(ave_reward)
    np.save('data\%s.npy'%test_name, np.array(plot_reward))
    plt.figure(1)
    plt.plot(plot_reward)
    plt.grid()
    plt.xlabel('episode')
    plt.ylabel('average_reward')
    plt.savefig('figure\%s'%test_name)
    plt.show()

if __name__ == '__main__':
    main()