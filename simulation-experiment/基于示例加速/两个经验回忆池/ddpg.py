# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------
import tensorflow as tf
import numpy as np
import pickle
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network_bn import ActorNetwork
from replay_buffer import ReplayBuffer,PrioritizedReplayBuffer

# Hyper Parameters:
REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 100000
BATCH_SIZE = 64
TRAIN_BATCH_SIZE=32
GAMMA = 0.99

class DDPG:
    """docstring for DDPG"""
    def __init__(self, env):
        self.name = 'DDPG' # name for uploading results
        self.environment = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.time_step=1
        self.save_network=True
        self.actor_load=False
        self.critic_load=False
        self.priority=True
        self.demo=True
        self.exploration_tatio=1
        self.sess=tf.Session()
        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim, self.actor_load)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim, self.critic_load,self.priority)

        # initialize replay buffer
        if self.priority:
            self.replay_buffer = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE,self.demo)
        else:
            self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE,self.demo)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):
        self.time_step+=1

        # Sample a random minibatch of N transitions from replay buffer

        demo_tree_idx, demo_minibatch, demo_ISWeights ,demo_pri= self.replay_buffer.demo_sample(TRAIN_BATCH_SIZE)
        intera_tree_idx, intera_minibatch, intera_ISWeights, intera_pri = self.replay_buffer.intera_sample(TRAIN_BATCH_SIZE)
        ISWeights=np.ones((TRAIN_BATCH_SIZE,1))


        demo_state_batch = np.asarray([data[0] for data in demo_minibatch])
        demo_action_batch = np.asarray([data[1] for data in demo_minibatch])
        demo_reward_batch = np.asarray([data[2] for data in demo_minibatch])
        demo_next_state_batch = np.asarray([data[3] for data in demo_minibatch])
        demo_done_batch = np.asarray([data[4] for data in demo_minibatch])

        intera_state_batch = np.asarray([data[0] for data in intera_minibatch])
        intera_action_batch = np.asarray([data[1] for data in intera_minibatch])
        intera_reward_batch = np.asarray([data[2] for data in intera_minibatch])
        intera_next_state_batch = np.asarray([data[3] for data in intera_minibatch])
        intera_done_batch = np.asarray([data[4] for data in intera_minibatch])

        # for action_dim = 1
        intera_action_batch = np.resize(intera_action_batch,[TRAIN_BATCH_SIZE,self.action_dim])
        demo_action_batch = np.resize(demo_action_batch, [TRAIN_BATCH_SIZE, self.action_dim])

        # Calculate demo_y_batch
        demo_next_action_batch = self.actor_network.target_actions(demo_next_state_batch)
        demo_q_value_batch = self.critic_network.target_q(demo_next_state_batch,demo_next_action_batch)
        demo_y_batch = []
        for i in range(len(demo_minibatch)):
            if demo_done_batch[i]:
                demo_y_batch.append(demo_reward_batch[i])
            else :
                demo_y_batch.append(demo_reward_batch[i] + GAMMA * demo_q_value_batch[i])
        demo_y_batch = np.resize(demo_y_batch,[TRAIN_BATCH_SIZE,1])

        # Calculate intera_y_batch
        intera_next_action_batch = self.actor_network.target_actions(intera_next_state_batch)
        intera_q_value_batch = self.critic_network.target_q(intera_next_state_batch, intera_next_action_batch)
        intera_y_batch = []
        for i in range(len(intera_minibatch)):
            if intera_done_batch[i]:
                intera_y_batch.append(intera_reward_batch[i])
            else:
                intera_y_batch.append(intera_reward_batch[i] + GAMMA * intera_q_value_batch[i])
        intera_y_batch = np.resize(intera_y_batch, [TRAIN_BATCH_SIZE, 1])

        # Update critic by minimizing the loss L
        demo_abs_errors,self.demo_cost=self.critic_network.train(demo_y_batch,demo_state_batch,demo_action_batch,ISWeights)
        intera_abs_errors, self.intera_cost = self.critic_network.train(intera_y_batch, intera_state_batch, intera_action_batch, ISWeights)
        self.replay_buffer.demo_batch_update(demo_tree_idx, demo_abs_errors) # update priority
        self.replay_buffer.intera_batch_update(intera_tree_idx, intera_abs_errors)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(intera_state_batch)
        q_gradient_batch = self.critic_network.gradients(intera_state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,intera_state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    #预训练是为了进行对优先级进行排序，在这个过程中对网络参数不进行更新
    def per_train(self):
        tree_idx, minibatch, ISWeights, pri = self.replay_buffer.demo_sample(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])
        # for action_dim = 1
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])
        # Calculate y_batch
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])
        abs_errors = self.critic_network.per_train(y_batch, state_batch, action_batch)
        self.replay_buffer.demo_batch_update(tree_idx, abs_errors)


    def per_add(self):
        with open('data\object.pickle', 'rb') as f:
            self.buffer = pickle.load(f)
            print("数据加载完成")
        for i in range(REPLAY_BUFFER_SIZE):
            state=self.buffer[i][0]
            action=self.buffer[i][1]
            reward=self.buffer[i][2]
            next_state=self.buffer[i][3]
            done=self.buffer[i][4]
            self.replay_buffer.demo_add(state, action, reward, next_state, done)
        print('数据添加完成')

    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        return action+self.exploration_tatio*self.exploration_noise.noise()

    def action(self,state):
        action = self.actor_network.action(state)
        return action

    def critic(self,state,action):
        critic=self.critic_network.single_q_value(state,action)
        return critic

    def perceive(self,state,action,reward,next_state,done):
        self.replay_buffer.intera_add(state, action, reward, next_state, done)
        if self.replay_buffer.intera_count()==REPLAY_START_SIZE:
            print('开始正式训练')
        if self.replay_buffer.intera_count() > REPLAY_START_SIZE:
            self.train()

        if (self.time_step+10000) % 100000 == 0:
            if self.save_network:
                self.actor_network.save_network(self.time_step,self.name)
                self.critic_network.save_network(self.time_step,self.name)

        # Re-initialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

    def reset(self):
        self.replay_buffer.erase()