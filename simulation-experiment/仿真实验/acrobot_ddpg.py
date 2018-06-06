import filter_env
import time
import gc
import matplotlib.pyplot as plt
import numpy as np
import gym
from ddpg import *
plt.rcParams['font.sans-serif']=['SimHei']#增加中文功能
plt.rcParams['axes.unicode_minus']=False
gc.enable()

# ENV_NAME = 'Acrobot-v1'
ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'MountainCarContinuous-v0'
test_name='示教实验9'
EPISODES = 2000
TEST = 1

plot_reward=[]
def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    for episode in range(EPISODES):
        # Train
        state = env.reset()
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
            for i in range(TEST):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    env.render()
                    action = agent.action(state)
                    state, reward, done, inf = env.step(action)
                    total_reward += reward
                    if done:
                         break
            ave_reward = total_reward/TEST
            print('episode: ', episode, 'Evaluation Average Reward:',
                ave_reward,time.strftime('  %Y-%m-%d %A %X',time.localtime()))
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