# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/1/17 19:17
'''

import filter_env
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network_bn import ActorNetwork
from buffer import ReplayBuffer,ReplayDemo
import gym
import time

PRE_BATCH_SIZE=64
H=np.ones([PRE_BATCH_SIZE,1])

class DDPG_DEMO_PRE(object):
    def __init__(self,env):
        self.name = 'DDPG_DEMO'  # name for uploading results
        self.pre_train_step = -1
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.save_network = True
        self.actor_load = False
        self.critic_load = False

        self.sess = tf.Session()
        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim, self.actor_load)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim, self.critic_load)
        self.replay_demo = ReplayDemo()

    def pre_train(self):
        self.pre_train_step+=1
        minibatch = self.replay_demo.get_batch(PRE_BATCH_SIZE)
        demo_state_batch = np.asarray([data[0] for data in minibatch])
        demo_action_batch = np.asarray([data[1] for data in minibatch])
        demo_action_batch = np.resize(demo_action_batch, [PRE_BATCH_SIZE, 1])

        #Calculate q_sup
        # action_batch=self.actor_network.target_actions(demo_state_batch)
        action_batch = self.actor_network.actions(demo_state_batch)
        cost=np.mean(np.square(10*(demo_action_batch-action_batch)))
        # q_sup_batch=H*(abs(action_batch-demo_action_batch)>0.1)+self.critic_network.target_q(demo_state_batch,action_batch)
        q_sup_batch = H * (abs(action_batch - demo_action_batch) > 0.1) + self.critic_network.q_value(demo_state_batch,
                                                                                                       action_batch)
        # print(H*(abs(action_batch-demo_action_batch)>0.01))
        # Update critic by minimizing the loss L
        q_sup_batch = np.resize(q_sup_batch, [PRE_BATCH_SIZE, 1])
        self.critic_network.train(q_sup_batch, demo_state_batch, demo_action_batch)
        # print(q_sup_batch[0],self.critic_network.q_value(demo_state_batch, demo_action_batch)[0])
        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(demo_state_batch)
        q_gradient_batch = self.critic_network.gradients(demo_state_batch, action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch, demo_state_batch)
        if self.pre_train_step%10000==0:
            print("动作：",demo_action_batch[0],action_batch[0])
            print("价值：",self.critic_network.target_q(demo_state_batch,action_batch)[0],self.critic_network.q_value(demo_state_batch,action_batch)[0])
            print("action_batch_for_gradients:",action_batch_for_gradients[0])
            print("q_gradient_batch:",q_gradient_batch[0])
        #update network
        self.actor_network.update_target()
        self.critic_network.update_target()
        return cost

    def save_net(self,time_step):
        #save network
        self.actor_network.save_network(time_step,self.name)
        self.critic_network.save_network(time_step,self.name)

    def action(self,state):
        action = self.actor_network.action(state)
        return action

if __name__=='__main__':
    epsiode=500000
    # ENV_NAME='Acrobot-v1'
    ENV_NAME = 'Pendulum-v0'
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    ddpg_demo_agent=DDPG_DEMO_PRE(env)

    #预训练
    for step in range(epsiode):
        cost=ddpg_demo_agent.pre_train()
        if step%10000==0:
            print("step:%s,cost:%s"%(step,cost))
    ddpg_demo_agent.save_net(epsiode)

    #测试
    state=env.reset()
    for i in range(1000):
        env.render()
        time.sleep(0.1)
        action=ddpg_demo_agent.action(state)
        next_state,reward,done,inf=env.step(action)
        state=next_state
        print(action)

        if done:
            break



