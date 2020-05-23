# 策略梯度算法  处理离散动作
# 2020.5.23
#
# pong state是210*160*3的RGB图像，action是3个，action 1: static, action 2: move up, action 3: move down
# reward 有三种：当丢球，reward为-1；接到球，reward为1；对手丢球，reward为0
# 最后是谁先得到21点的分数就结束

import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch policy gradient example at openai-gym pong')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
                    help='decay rate for RMSprop (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=20, metavar='G',
                    help='Every how many episodes to da a param update')
parser.add_argument('--seed', type=int, default=87, metavar='N',
                    help='random seed (default: 87)')
parser.add_argument('--test', action='store_false',
        help='whether to test the trained model or keep training')

args = parser.parse_args()

test = args.test
if test == True:
    render = True
else:
    render = False

env = gym.make('Pong-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
print('state space:',env.observation_space)
print('action space:',env.action_space)

D=80*80  #6400

def prepross(I):
    """prepross 210*160*3 into 6400"""
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


class Policy(nn.Module):
    def __init__(self,num_actions=2):
        super(Policy,self).__init__()
        self.affline1 = nn.Linear(6400,200)
        self.affline2 = nn.Linear(200,num_actions)

        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []


    def forward(self, x):
        x = self.affline1(x)
        x = F.relu(x)
        action_scores = self.affline2(x)
        return F.softmax(action_scores, dim=1)

    def select_action(self,x):
        state = torch.from_numpy(x).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action


policy = Policy()
optimizer = optim.RMSprop(policy.parameters(), lr=args.learning_rate,weight_decay=args.decay_rate)

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma*R
        rewards.insert(0,R)
    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    for log_prob, reward in zip(policy.saved_log_probs,rewards):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = None
    reward_sum = 0
    prev_x = None
    for i_episode in range(1,1000):
        state = env.reset()
        for t in range(10000):
            if render:
                env.render()
            cur_x = prepross(state)
            x = cur_x - prev_x if prev_x is not None else np.zeros(D)  # 几帧之差
            prev_x = cur_x
            action = policy.select_action(x)
            action_env = action + 2              #
            state, reward, done,_ = env.step(action_env)
            reward_sum += reward

            policy.rewards.append(reward)
            if done:
                # tracking log 追踪记录
                running_reward = reward_sum if running_reward is None else running_reward*0.99+reward_sum*0.01
                print('REINFORCE ep %03d done. reward: %f. reward running mean: %f' % (
                i_episode, reward_sum, running_reward))
                reward_sum = 0
                break


            # use policy gradient update model weights
        if i_episode % args.batch_size == 0:
            finish_episode()

            # Save model in every 50 episode
        if i_episode % 50 == 0:
            print('ep %d: model saving...' % (i_episode))
            torch.save(policy.state_dict(), 'pg_params.pkl')

if __name__ == '__main__':
    main()
