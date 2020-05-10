#

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import gym
import numpy as np
import random
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    # state_size (int): Dimension of each state
    # action_size (int): Dimension of each action
    # seed (int): Random seed
    def __init__(self, state_size, action_size, seed):
        super(QNetwork,self).__init__()

        self.seed = torch.manual_seed(seed)
        self.conv = nn.Sequential(
            nn.Conv2d(state_size[1],32,8,4),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, state):
        conv_out = self.conv(state).view(state.size()[0], -1)  # Faltten
        return self.fc(conv_out)


def pre_process(observation):
    # 将（210,160,3）转换为 （1,84,84）
    x_t = cv2.cvtColor(cv2.resize(observation,(84,84)), cv2.COLOR_BGR2GRAY) # 裁剪并灰度化
    ret, x_t = cv2.threshold(x_t, thresh=1,maxval=255, type=cv2.THRESH_BINARY)    # 二值化
    return np.reshape(x_t, (1,84,84)), x_t


def stack_state(processed_obs):
    # 将连续的四张图片作为一个state
    return np.stack((processed_obs, processed_obs, processed_obs, processed_obs), axis=0)






if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    print('State space: ',env.observation_space.shape)    # (210, 160, 3)
    print('Number of Actions: ',env.action_space.n)    # 动作有四种，分别为向左、右、不动和发射

    obs = env.reset()
    x_t, img = pre_process(obs)
    state = stack_state(img)
    print(np.shape(state[0]))   # (84,84)

    # plt.imshow(img, cmap='gray')
    # cv2.imshow('Breakout', img)
    # cv2.waitKey(0)

    state = torch.randn(32,4,84,84)   # (batch_size, color_channel, img_height,img_width)
    state_size = state.size()
    print(state)
    #print(state_size[1]) #4

    cnn_model = QNetwork(state_size, action_size=4, seed=1)
    outputs = cnn_model(state)
    print(outputs.shape)


