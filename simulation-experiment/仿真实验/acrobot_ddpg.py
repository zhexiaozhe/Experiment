import filter_env
from ddpg import *
import time
from numpy import pi
# gc is memory manage modle
import gc
import matplotlib.pyplot as plt
gc.enable()
import numpy as np

import itchat

# ENV_NAME='MountainCarContinuous-v0'
ENV_NAME = 'Acrobot-v1'
test_name='test 75'
EPISODES = 3000
TEST = 1
#单目标
plot_reward=[]
plot_step=[]

def main():
    #登录微信,登录时将生成.pkl文件这个文件保存的是登录信息
    ###################################################
    itchat.auto_login(hotReload=True)
    ###################################################
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    for episode in range(EPISODES):
        state = env.reset()
        # Train
        for step in range(env.spec.timestep_limit): #Pendulum-v0:env.spec.timestep_limit=200
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done) #if env = = cartpole need next_state flatten
            state = next_state
            if done:
                break

        # Testing:
        if (episode+1) % 50 == 0 or episode==0:
        # if episode % 50 == 0 and episode>0:
            total_reward=0
            testing_step = 0

            for i in range(TEST):
                state = env.reset()
                theta2=[]
                theta1=[]
                theta2d=[]
                theta1d=[]
                Y=[]
                dtheta1=[]
                dtheta2 = []
                energy=[]
                Ed=[]
                step=[]
                R=[]
                action_figure=[]
                reward_error=[]
                distance=[]
                PHI=[]
                R1=[]
                R2=[]
                Action=[]
                Action_smoothing=[]
                for j in range(env.spec.timestep_limit):
                    env.render()

                    # start_time=time.clock()
                    # if j<=100:
                    #      action=0.4
                    # else:
                    #      action = agent.action(state) # direct action for test
                    action = agent.action(state)
                    if j==0:
                        action_smoothing=action
                    action_smoothing=0.2*action+0.8*action_smoothing
                    # print(time.clock()-start_time)
                    # critic_value=agent.critic(state,action)[0]
                    state,reward,done,inf = env.step(action_smoothing)

                    #虚约束与能量结合
                    # PHI.append(inf[3])
                    # theta1.append(inf[2][0])
                    # theta2.append(inf[2][1])
                    # theta1d.append(-pi/4)
                    # theta2d.append(pi/4)
                    # Ed.append(inf[6])
                    # energy.append(inf[5])
                    # Action.append(action)
                    # Action_smoothing.append(action_smoothing)
                    # R1.append(5*inf[0])
                    # R2.append(inf[1])
                    # distance.append(inf[4])
                    # dtheta1.append(inf[2][2])
                    # dtheta2.append(inf[2][3])
                    # step.append(j)

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
                    # step.append(j)
                    # theta2.append([inf[1][0],inf[1][1]])
                    # energy.append(inf[0])
                    # Ed.append(inf[2])
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
                #虚约束与能量结合
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
                #
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
                #pendulum
                # plt.figure(1)
                # plt.subplot(211)
                # plt.plot(theta)
                # plt.ylabel('angle/rad')
                # plt.xlabel('step')
                # plt.subplot(212)
                # plt.plot(dtheta)
                # plt.ylabel('speed/rad.s-1')
                # plt.xlabel('step')
                # plt.show()

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
            # ave_step=testing_step/TEST
            # print('episode: ',episode,'key=',key,'Evaluation Average Reward:',ave_reward)
            print('episode: ', episode, 'Evaluation Average Reward:',
                ave_reward,time.strftime('  %Y-%m-%d %A %X',time.localtime()))
            itchat.send("episode:"+str(episode)+",Evaluation Average Reward:"+str(ave_reward))
            #单目标
            plot_reward.append(ave_reward)
            # plot_step.append(ave_step)
    np.save('data\ex75.npy', np.array(plot_reward))

    plt.figure(1)
    plt.plot(plot_reward)
    plt.grid()
    plt.xlabel('step')
    plt.ylabel('average_reward')
    plt.savefig('figure\%s'%test_name)
    time.sleep(10)
    itchat.send_image('figure\%s.png'%test_name)
    time.sleep(10)
    itchat.logout()
    plt.show()
if __name__ == '__main__':
    main()