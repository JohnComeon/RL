# ppo 算法实现，针对cartpole
# 2020.05.31

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

EPISODES = 1000
lr = 0.001
gamma = 0.98
lmbda = 0.95
epochs = 3
eps_clip = 0.2
MAX_STEPS = 1000


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 256)

        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def pi(self, x, softmax_dim=0):  # actor  policy approximation
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):  # critic   value approximation
        x = F.relu(self.fc1(x))
        x = self.fc_v(x)
        return x

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, next_s_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for t in self.data:
            s, a, r, next_s, prob_a, done = t
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            next_s_lst.append(next_s)
            prob_a_lst.append([prob_a])  # prob_a 是一个list
            done_mask = 0 if done else 1  # 二值化
            done_lst.append([done_mask])
        s, a, r, next_s, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                             torch.tensor(r_lst), torch.tensor(next_s_lst, dtype=torch.float), \
                                             torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, next_s, done_mask, prob_a

    def train_net(self):
        s, a, r, next_s, done_mask, prob_a = self.make_batch()

        for i in range(epochs):
            td_target = r + gamma * self.v(next_s) * done_mask
            delta = td_target - self.v(s)   # TD_error
            delta = delta.detach().numpy()         # 不计算梯度

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]   # gae
                advantage_lst.append(advantage)
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)  # 从pi的每一行提取出动作a的概率
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(surr1, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2).mean() + F.mse_loss(self.v(s), td_target.detach())
                         #  actor_loss +  critic_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    model = PPO()
    score = 0.0
    S = [0]
    X = [0]
    print_interval = 20

    for n_epi in range(EPISODES):
        s = env.reset()
        done = False
        while not done:
            for t in range(MAX_STEPS):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample()
                a = a.item()
                next_s, r, done, _ = env.step(a)
                model.put_data((s, a, r / 100, next_s, prob[a].item(), done))
                s = next_s
                score += r

                if done:
                    break

            model.train_net()


        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            S.append(score/print_interval)
            X.append(n_epi)
            score = 0
    plt.plot(X,S, '-')
    plt.legend(['PPO'])
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('demo')
    plt.pause(5)
    env.close()

if __name__ == "__main__":
    main()
    # a = torch.rand(3,4,5)
    # print(a)
    # b = F.softmax(a,dim=0)
    # print(b)
