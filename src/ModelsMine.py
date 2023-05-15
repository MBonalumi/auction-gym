#################################################################
##### https://github.com/pranz24/pytorch-soft-actor-critic  #####
#################################################################
# TODO: citare la repo
# ora è chatgpt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer_1(state))
        x = torch.relu(self.layer_2(x))

        ### MODIFIED  ->  i want actions bw (0, max_action] 
        x = self.max_action * (1 + torch.tanh(self.layer_3(x))) / 2
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim + action_dim, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 1)

    def forward(self, state, action):
        pass
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

class ReplayBuffer(object):
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indexes = np.random.choice(len(self.buffer), batch_size, replace=False) # replace=False -> no duplicates
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in indexes])
        pass
        # state, action, reward, next_state, done = zip(*np.random.choice(self.buffer, batch_size, replace=False)) #bad solution resulting in error: 'a must be 1-dimensional'
        return np.array(state), np.array(action), np.array(reward).reshape(-1, 1), np.array(next_state), np.array(done).reshape(-1, 1)

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(1000000)
        self.batch_size = 256
        self.discount = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, iterations):
        for it in range(iterations):
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

            state = torch.FloatTensor(state)
            action = torch.FloatTensor(action)
            action = action[:,None]
            reward = torch.FloatTensor(reward)
            next_state = torch.FloatTensor(next_state)
            done = torch.FloatTensor(done)

            pass

            next_action = self.actor(next_state)
            noise = torch.randn_like(next_action) * 0.2
            next_action = (next_action + noise).clamp(-self.actor.max_action, self.actor.max_action)

            target_Q1 = self.critic_1(next_state, next_action)
            target_Q2 = self.critic_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.discount * target_Q).detach()

            current_Q1 = self.critic_1(state, action)
            current_Q2 = self.critic_2(state, action)

            critic_1_loss = nn.MSELoss()(current_Q1, target_Q)
            critic_2_loss = nn.MSELoss()(current_Q2, target_Q)

            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()

            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            if it % 2 == 0:
                actor_loss = -(self.critic_1(state, self.actor(state)).mean())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic_1.parameters(), self.critic_2.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                # WHY DO THIS? ### should there be a TARGET ACTOR ??? ###
                for param, target_param in zip(self.actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def save(self, filename):
    torch.save(self.actor.state_dict(), filename)

def load(self, filename):
    self.actor.load_state_dict(torch.load(filename))
    self.actor_target.load_state_dict(torch.load(filename))

