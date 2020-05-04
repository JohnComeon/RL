## Sarsa算法实现
# 2020.5.3

import gym
from gym import Env
from random import random
from gridworld import *

class Agent():
    def __init__(self, env: Env):
        self.env = env
        self.Q = {}
        self._initAgent()
        self.state = None

    def performPolicy(self, s, episode_num, use_epsilon):   # 执行一个策略
        # use_epsilon参数用于判断是否要使用e-greedy
        epsilon = 1.00 / (episode_num + 1)
        Q_s = self.Q[s]
        str_act = 'unknown'
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:
            action = self.env.action_space.sample()  # 生成随机行为
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action

    def act(self, a):   # 执行一个动作
        return self.env.step(a)

    def _get_state_name(self, state):   # 得到状态对应的字符串作为以字典存储的值函数的键
        return str(state)              # 应该针对不同的状态值单独设计，这里仅针对格子世界

    def _is_state_in_Q(self,s):  # 判断s的Q值是否存在
        return self.Q.get(s) is not None

    def _init_state_value(self, s_name, randomized=True):  # 初始化某状态的Q值
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.env.action_space.n):
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v

    def _assert_state_in_Q(self, s, randomized=True):  # 确保某状态Q值存在
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)

    def _get_Q(self, s, a):  # 获取Q(s,a)
        self._assert_state_in_Q(s, randomized=True)
        return self.Q[s][a]

    def _set_Q(self,s, a, value):
        self._assert_state_in_Q(s, randomized=True)
        self.Q[s][a] = value

    def _initAgent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_Q(s_name, randomized=False)

    def learning(self, gamma, alpha, max_episode_num):
        # Sarsa learning
        total_time, time_in_episode, num_episodes = 0, 0, 0
        while num_episodes < max_episode_num:
            self.state = self.env.reset()   # 环境初始化
            s0 = self._get_state_name(self.state)   # 获取个体对于环境观测的命名
            self.env.render()  # 渲染环境，显示UI界面
            a0 = self.performPolicy(s0, num_episodes, use_epsilon=True)

            time_in_episode = 0
            is_done = False
            while not is_done:
                s1, r1, is_done, info = self.act(a0)   # 执行行为
                self.env.render()
                s1 = self._get_state_name(s1)
                self._assert_state_in_Q(s1, randomized=True)
                a1 = self.performPolicy(s1, num_episodes, use_epsilon=True)
                old_q = self._get_Q(s0, a0)
                q_prime = self._get_Q(s1, a1)   # Q'
                td_target = r1 + gamma*q_prime
                new_q = old_q + alpha * (td_target - old_q)
                self._set_Q(s0, a0, new_q)

                if num_episodes == max_episode_num:
                    print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}".format(time_in_episode, s0, a0, s1))
                s0, a0 = s1, a1
                time_in_episode += 1
            self.env.close()
            print("Episode {0} takes {1} steps.".format(num_episodes, time_in_episode))
            total_time += time_in_episode
            num_episodes += 1

        return


def main():
    env = SimpleGridWorld()
    agent = Agent(env)
    env.reset()
    print('learning...')
    agent.learning(gamma=0.9, alpha=0.1, max_episode_num=800)
    env.close()


if __name__ == "__main__":
    main()
