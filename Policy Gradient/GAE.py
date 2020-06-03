#coding:utf-8
# Generalized Advantage Estimation  "Pendulum-v0"
# 2020.06.01


import math
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

from common.multiprocessing_env import SubprocVecEnv


env_name = "Pendulum-v0"

env = gym.make(env_name)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic,self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)      # 均值μ
        std = self.log_std.exp().expand_as(mu)    # 标准差σ
        dist = Normal(mu, std)
        return dist, value

def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist,_ = model(state)
        next_state, reward, done,_ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def compute_gae(next_value,rewards, done_mask, values, gamma=0.99, labda=0.95):
    values = values + [next_value]
    gae = []
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * done_mask[step] - values[step]   # δ
        gae = delta + gamma * labda * done_mask[step] * gae             # 求和
        returns.insert(0, gae+values[step])
    return returns

if __name__ == '__main__':
    num_envs = 8
    def make_env():
        def _thunk():
            env = gym.make(env_name)
            return env
        return _thunk
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(env_name)
    num_inputs = envs.observation_space.shape[0]
    num_outputs = envs.action_space.shape[0]

    hidden_size = 256
    lr = 3e-2
    num_steps = 20

    model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(),lr)

    max_frames = 100000
    frame_idx = 0
    test_rewards = []

    state = envs.reset()

    while frame_idx < max_frames:
        log_probs = []
        values = []
        rewards = []
        done_mask = []
        entropy = 0

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action)

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1)).to(device)
            done_mask.append(torch.FloatTensor(1 - done).unsqueeze(1)).to(device)

            state = next_state
            frame_idx += 1

            if frame_idx % 1000 == 0:
                test_rewards.append(np.mean([test_env() for _ in range(10)]))
                plot(frame_idx, test_rewards)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards,done_mask,values)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5*critic_loss - 0.001*entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#test_env(True)

#     main()