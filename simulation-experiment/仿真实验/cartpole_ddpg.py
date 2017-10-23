import filter_env
from ddpg import *
import time
# gc is memory manage modle
import gc
import matplotlib.pyplot as plt
gc.enable()
import numpy as np
#ENV_NAME = 'InvertedPendulum-v1'
#ENV_NAME ='Pendulum-v0'
ENV_NAME = 'CartPole-v0'
#ENV_NAME = 'Acrobot-v1'
test_name='test 20d'
EPISODES = 50000
TEST = 10
MAX_EP_STEP=2000
plot_reward=[]
plot_step=[]
def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))

    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)

    for episode in range(EPISODES):

        #print("episode:",episode)
        state = env.reset()

        # Train
        for step in range(env.spec.timestep_limit): #Pendulum-v0:env.spec.timestep_limit=200
        #for step in range(MAX_EP_STEP):
            #env.render()
            #time.sleep(5)
            #print(state)
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            #print(next_state)
            #print(reward)
            #print(2*action)
            #print(step)
            #if reward!=0:
                #print('reward=%s !!!!!!!!!!!!!!!!!!!!!',reward)
            agent.perceive(state,action,reward,next_state,done)
            #done=True if step == MAX_EP_STEP - 1 else False
            state = next_state

            if done:
                break

        # Testing:
        if episode % 100 == 0 and episode>0:
            total_reward=0
            testing_step = 0
            for i in range(TEST):
                state = env.reset()

                for j in range(MAX_EP_STEP):
                #for j in range(MAX_EP_STEP):
                #while True:
                   # if episode>80000:
                   env.render()
                   action = agent.action(state) # direct action for test

                   state,reward,done,_ = env.step(action)
                   #print("state:%s ep:%s a:%s r:%s" % (state,j, action,reward))
                   # if j==400:
                   #     print('done')
                   done = True if j ==  - 1 else False
                   total_reward += reward

                   #if j==5000:
                       #done=True
                       #print("finish good!!!!!!!!!!!!")
                   testing_step+=1
                   if done:
                        break
            ave_reward = total_reward/TEST
            ave_step=testing_step/TEST
            print('episode: ',episode,'Evaluation Average Reward:',ave_reward,'Average Step:',ave_step)
            #print('episode: ', episode, 'Evaluation Average step:', ave_step)
        #env.monitor.close()
            plot_reward.append(ave_reward)
            #plot_step.append(ave_step)

    plt.plot(plot_reward)
    plt.xlabel('step')
    plt.ylabel('average_reward')
    plt.savefig('test_name：%s DDPG_%s step_%s'%(test_name,ENV_NAME,EPISODES))
    plt.show()

if __name__ == '__main__':
    main()
