# -*- coding: utf-8 -*-
'''
@author: 程哲
@contact: 909991719@qq.com
@time: 2018/10/19 20:59
'''
#加载gym模块
import gym
#创建仿真环境
env = gym.make('Acrobot-v1')
#主循环
for i_episode in range(20):
    #初始化仿真环境
    observation = env.reset()
    #开始一个episode
    for t in range(100):
        #可视化
        env.render()
        #智能体决策
        action = env.action_space.sample()
        #智能体与仿真环境交互
        observation, reward, done, info = env.step(action)
        #判断一个episode是否结束
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
