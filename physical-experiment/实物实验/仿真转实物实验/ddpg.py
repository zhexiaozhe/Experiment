# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------
import gym
import math
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network_bn import ActorNetwork
from replay_buffer import ReplayBuffer

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99

class DDPG:
    """docstring for DDPG"""
    def __init__(self):
        self.name = 'DDPG' # name for uploading results
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = 6
        self.action_dim = 1
        self.time_step=1
        self.save_network=True
        self.actor_load=True
        self.critic_load=True
        self.exploration_tatio=1
        # self.actor_sess = tf.Session()
        # self.critic_sess=tf.Session()
        self.sess=tf.Session()
        # 以下两个部分谁在后面谁就保存了整个网络参数
        # 网络整体保存
        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim, self.actor_load)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim, self.critic_load)

        #网络分开保存
        # self.actor_network = ActorNetwork(self.actor_sess, self.state_dim, self.action_dim, self.actor_load)
        # self.critic_network = CriticNetwork(self.critic_sess, self.state_dim, self.action_dim, self.critic_load)



        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):
        self.time_step+=1

        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE) #原来代码
        # tree_idx, minibatch, ISWeights = self.replay_buffer.sample(BATCH_SIZE)

        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch
        
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []
        for i in range(len(minibatch)): 
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
            #print(reward_batch[i],q_value_batch[i])
        #y_batch是目标值
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        # abs_errors,self.cost=self.critic_network.train(ISWeights,y_batch,state_batch,action_batch)
        self.critic_network.train(y_batch, state_batch, action_batch)
        # self.replay_buffer.batch_update(tree_idx, abs_errors)  # update priority

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        # if self.time_step%2500000==0:
        #     self.exploration_tatio/=2
        return action+self.exploration_tatio*self.exploration_noise.noise()

    def action(self,state):
        action = self.actor_network.action(state)
        return action

    def critic(self,state,action):
        critic=self.critic_network.single_q_value(state,action)
        return critic

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            self.train()

        if (self.time_step+10000) % 100000 == 0:
            if self.save_network:
                self.actor_network.save_network(self.time_step,self.name)
                self.critic_network.save_network(self.time_step,self.name)

        # Re-initialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()










