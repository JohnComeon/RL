#

import numpy as np
import random
from collections import deque,namedtuple
import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import QNetwork

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 32
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-5               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# 经验回放
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
                Params
                ======
                    action_size (int): dimension of each action
                    buffer_size (int): maximum size of buffer
                    batch_size (int): size of each training batch
                    seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience",field_names=["state", "action", "reward", "next_state", "done"])

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):      # 随机采样
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states,actions,rewards,next_states,dones)

    def __len__(self):
        return len(self.memory)


class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)   # behavior network
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)  # target network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0     # Initialize time step (for updating every UPDATE_EVERY steps)

    def step(self, state, action, reward, next_state, done):
    # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY  # Learn every UPDATE_EVERY time steps.达到的固定步数才去学习
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:   # If enough samples are available in memory, get random subset and learn
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)     # 更新网络

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
                Params
                ======
                    state (array_like): current state
                    eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
                θ_target = τ*θ_local + (1 - τ)*θ_target
                Params
                ======
                    local_model (PyTorch model): weights will be copied from
                    target_model (PyTorch model): weights will be copied to
                    tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
                Params
                ======
                    experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
                    gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Get max predicted Q values (for next states) from target model
        # 在Double DQN中，Q_targets_next变成了self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        #print(self.qnetwork_local(states).shape)

        Q_expected = self.qnetwork_local(states).gather(1, actions)   # 固定行号，确认行号；找到使得Q最大的action

        # 计算 loss
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


