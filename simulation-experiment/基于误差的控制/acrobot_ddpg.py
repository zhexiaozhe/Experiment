import filter_env
from ddpg import *
import time
from numpy import pi
# gc is memory manage modle
import gc
import matplotlib.pyplot as plt
import numpy as np
import itchat
import gym
gc.enable()

ENV_NAME = 'Acrobot-v1'
test_name='experiment 2'
EPISODES = 1000
TEST = 1

#单目标
plot_reward=[]
plot_step=[]

def main():
    #登录微信,登录时将生成.pkl文件这个文件保存的是登录信息
    ###################################################
    # itchat.auto_login(hotReload=True)
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
                Tau_dtheta1=[]
                Tau_dtheta2 = []
                Action=[]
                Action_smoothing=[]
                Q_value=[]
                for j in range(env.spec.timestep_limit):
                    env.render()
                    # start_time=time.clock()
                    # if j<=100:
                    #      action=0.4
                    # else:
                    #      action = agent.action(state) # direct action for test
                    # if j<=200:
                    #     action=0.5
                    # else:
                    action = agent.action(state)
                    if j==0:
                        action_smoothing=action
                    action_smoothing=1*action+0.0*action_smoothing
                    # critic_value=agent.critic(state,action)[0]
                    state,reward,done,inf = env.step(action_smoothing)
                    # print(inf[0],inf[1])
                    # print(inf[6],inf[7])

                    total_reward += reward
                    if done:
                         break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:',
                ave_reward,time.strftime('  %Y-%m-%d %A %X',time.localtime()))
            # itchat.send("episode:"+str(episode)+",Evaluation Average Reward:"+str(ave_reward))
            #单目标
            plot_reward.append(ave_reward)
            # plot_step.append(ave_step)
    np.save('data\%s.npy'%test_name, np.array(plot_reward))
    plt.figure(1)
    plt.plot(plot_reward)
    plt.grid()
    plt.xlabel('step')
    plt.ylabel('average_reward')
    plt.savefig('figure\%s'%test_name)
    # time.sleep(10)
    # itchat.send_image('figure\%s.png'%test_name)
    # time.sleep(10)
    # itchat.logout()
    plt.show()
if __name__ == '__main__':
    main()