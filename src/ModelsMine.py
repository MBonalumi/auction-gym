#################################################################
##### https://github.com/pranz24/pytorch-soft-actor-critic  #####
#################################################################

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
    def __init__(self, max_size, rng):
        self.rng = rng
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        indexes = self.rng.choice(len(self.buffer), batch_size, replace=False) # replace=False -> no duplicates
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in indexes])
        pass
        # state, action, reward, next_state, done = zip(*self.rng.choice(self.buffer, batch_size, replace=False)) #bad solution resulting in error: 'a must be 1-dimensional'
        return np.array(state), np.array(action), np.array(reward).reshape(-1, 1), np.array(next_state), np.array(done).reshape(-1, 1)

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, rng=None):
        self.rng = rng

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(1000000, self.rng)
        self.batch_size = 256
        self.gamma = gamma  # discount factor
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

            next_action = self.actor(next_state)
            noise = torch.randn_like(next_action) * 0.2
            next_action = (next_action + noise).clamp(-self.actor.max_action, self.actor.max_action)

            target_Q1 = self.critic_1(next_state, next_action)
            target_Q2 = self.critic_2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # with gamma=0 --> target_Q = reward
            # and i ignore (s', a') values/exp.reward
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

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


####################################
#####     SB3 Environment      #####
####################################

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BidEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, rng, num_bids):
        super().__init__()
        self.num_envs = 1       # TODO: should be useless, check
        self.rng = rng
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(num_bids)
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.actions_rewards = []
        self.observations = []

    def step(self, action):
        # get the reward at random from self.actions_rewards, matching the action
        actions_rewards = np.array(self.actions_rewards)
        observations = np.array(self.observations)

        # all the rewards when playing that action
        rewards = actions_rewards[:, actions_rewards[0] == action] [ 1 ]  # [1] to get the rewards

        observation = self.rng.choice(observations) # next observation is irrelevant (not learning state-transition)
        reward = self.rng.choice(rewards) if rewards else 0.0   
        # reward = rewards.mean() if rewards else 0.0   
        '''
        why a random reward? isn't mean better?
            no, since we want to learn the distribution of rewards,
            mean would always be the same
        '''
        
        #trying
        terminated = False
        truncated = False
        info = dict()   # {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset()
        #not sure what this should do
        # should actually never be called since terminated/truncated are always False
        observation = self.rng.choice(self.observations)
        info=dict()
        return observation, info

    def render(self):
        pass

    def close(self):
        pass



######################################################################################
##### https://github.com/Bigpig4396/Incremental-Gaussian-Process-Regression-IGPR #####
######################################################################################

### Incremental Gaussian Process Regression (IGPR) ###

import numpy as np
import csv
from collections import deque
import random

class HyperParam(object):
    def __init__(self, theta_f=1, len=1, theta_n=0.1):
        self.theta_f = theta_f       # for squared exponential kernel
        self.len = len           # for squared exponential kernel
        self.theta_n = theta_n     # for squared exponential kernel

class IGPR(object):
    def __init__(self, init_x, init_y):
        self.hyperparam = HyperParam(1, 1, 0.1)
        self.max_k_matrix_size = 1e6
        self.lamda = 1
        self.count = 0
        self.kernel_x = deque()
        self.kernel_y = deque()
        self.kernel_x.append(init_x)
        self.kernel_y.append(init_y)
        self.k_matrix = np.ones((1, 1)) + self.hyperparam.theta_n * self.hyperparam.theta_n
        self.inv_k_matrix = np.ones((1, 1)) / (self.hyperparam.theta_n * self.hyperparam.theta_n)
        self.is_av = False
        temp = np.sum(self.k_matrix, axis=0)
        self.delta = deque()
        for i in range(temp.shape[0]):
            self.delta.append(temp[i])

    def is_available(self):
        n = len(self.kernel_x)
        if n >= 2:
            self.is_av = True
        return self.is_av

    # def load_csv(self, file_name):
    #     with open(file_name, "r") as f:
    #         reader = csv.reader(f)
    #         columns = [row for row in reader]
    #     columns = np.array(columns)
    #     m_x, n_x = columns.shape
    #     data_set = np.zeros((m_x,n_x))
    #     for i in range(m_x):
    #         for j in range(n_x):
    #             data_set[i][j] = float(columns[i][j])
    #     return data_set

    def learn(self, new_x, new_y):
        for i in range(len(self.delta)):
            self.delta[i] = self.delta[i]*self.lamda

        if self.is_available():
            # print('available')
            if len(self.kernel_x) < self.max_k_matrix_size:
                # print('aug_update_SE_kernel')
                self.aug_update_SE_kernel(new_x, new_y)
            else:
                new_delta = self.count_delta(new_x)
                max_value, max_index = self.get_max(self.delta)
                if new_delta < max_value:
                    # self.schur_update_SE_kernel(new_x, new_y)
                    # print('SM_update_SE_kernel')
                    self.SM_update_SE_kernel(new_x, new_y, max_index)
                    self.count = self.count + 1
                    if self.count > 100:
                        self.count = 0
                        self.calculate_SE_kernel()
                        self.inv_k_matrix = np.linalg.inv(self.k_matrix)


        else:
            self.kernel_x.append(new_x)
            self.kernel_y.append(new_y)
            self.calculate_SE_kernel()
            self.inv_k_matrix = np.linalg.inv(self.k_matrix)

    def calculate_SE_kernel(self):
        n = len(self.kernel_x)
        self.k_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.k_matrix[i][j] = np.sum(np.square(self.kernel_x[i] - self.kernel_x[j]))
                self.k_matrix[i][j] = self.k_matrix[i][j] / (-2)
                self.k_matrix[i][j] = self.k_matrix[i][j] / self.hyperparam.len
                self.k_matrix[i][j] = self.k_matrix[i][j] / self.hyperparam.len
                self.k_matrix[i][j] = np.exp(self.k_matrix[i][j])
                self.k_matrix[i][j] = self.k_matrix[i][j] * self.hyperparam.theta_f
                self.k_matrix[i][j] = self.k_matrix[i][j] * self.hyperparam.theta_f
        self.k_matrix = self.k_matrix + self.hyperparam.theta_n * self.hyperparam.theta_n * np.eye(n)
        temp = np.sum(self.k_matrix, axis=0)
        self.delta = deque()
        for i in range(temp.shape[0]):
            self.delta.append(temp[i])

    def predict(self, coming_x):
        if self.is_available():
            n = len(self.kernel_x)
            cross_kernel_k = np.zeros((1, n))
            for i in range(n):
                cross_kernel_k[0, i] = np.sum(np.square(self.kernel_x[i] - coming_x))
                cross_kernel_k[0, i] = cross_kernel_k[0, i] / (-2)
                cross_kernel_k[0, i] = cross_kernel_k[0, i] / self.hyperparam.len
                cross_kernel_k[0, i] = cross_kernel_k[0, i] / self.hyperparam.len
                cross_kernel_k[0, i] = np.exp(cross_kernel_k[0, i])
                cross_kernel_k[0, i] = cross_kernel_k[0, i] * self.hyperparam.theta_f
                cross_kernel_k[0, i] = cross_kernel_k[0, i] * self.hyperparam.theta_f
            kernel_y_mat = self.kernel_y[0]
            for i in range(1, n):
                kernel_y_mat = np.vstack((kernel_y_mat, self.kernel_y[i]))
            # print('kernel_y',self.kernel_y)
            # print('kernel_y_mat', kernel_y_mat)
            # prediction = cross_kernel_k.dot(self.inv_k_matrix.dot(kernel_y_mat))
            prediction = cross_kernel_k.dot(self.inv_k_matrix.dot(kernel_y_mat))[0][0]
        else:
            prediction = self.kernel_y[0]
        return prediction

    def aug_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)
        self.kernel_x.append(new_x)
        self.kernel_y.append(new_y)
        self.k_matrix = np.hstack((self.k_matrix, np.zeros((n, 1))))
        self.k_matrix = np.vstack((self.k_matrix, np.zeros((1, n+1))))

        for i in range(n+1):
            self.k_matrix[i, n] = np.sum(np.square(self.kernel_x[i] - new_x))
            self.k_matrix[i, n] = self.k_matrix[i, n] / (-2)
            self.k_matrix[i, n] = self.k_matrix[i, n] / self.hyperparam.len
            self.k_matrix[i, n] = self.k_matrix[i, n] / self.hyperparam.len
            self.k_matrix[i, n] = np.exp(self.k_matrix[i, n])
            self.k_matrix[i, n] = self.k_matrix[i, n] * self.hyperparam.theta_f
            self.k_matrix[i, n] = self.k_matrix[i, n] * self.hyperparam.theta_f

        self.k_matrix[n, n] = self.k_matrix[n, n] + self.hyperparam.theta_n * self.hyperparam.theta_n
        self.k_matrix[n, 0:n] = (self.k_matrix[0:n, n]).T
        b = self.k_matrix[0:n, n].reshape((n, 1))
        # print('b', b)
        d = self.k_matrix[n, n]
        # print('d', d)
        e = self.inv_k_matrix.dot(b)
        # print('e', e)
        g = 1 / (d - (b.T).dot(e))
        # print('g', g)
        haha_11 = self.inv_k_matrix + g[0][0]*e.dot(e.T)
        haha_12 = -g[0][0]*e
        haha_21 = -g[0][0]*(e.T)
        haha_22 = g


        temp_1 = np.hstack((haha_11, haha_12))
        temp_2 = np.hstack((haha_21, haha_22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))

        # udpate delta
        for i in range(n):
            self.delta[i] = self.delta[i] + self.k_matrix[i, n]
        self.delta.append(0)

        for i in range(n+1):
            self.delta[n] = self.delta[n] + self.k_matrix[i, n]

    def schur_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)

        self.kernel_x.append(new_x)
        self.kernel_y.append(new_y)
        self.kernel_x.popleft()
        self.kernel_y.popleft()

        K2 = np.zeros((n, n))
        K2[0:n-1, 0:n-1] = self.k_matrix[1:n, 1:n]
        for i in range(n):
            K2[i, n-1] = np.sum(np.square(self.kernel_x[i] - new_x))
            K2[i, n-1] = K2[i, n-1] / (-2)
            K2[i, n-1] = K2[i, n-1] / self.hyperparam.len
            K2[i, n-1] = K2[i, n-1] / self.hyperparam.len
            K2[i, n-1] = np.exp(K2[i, n-1])
            K2[i, n-1] = K2[i, n-1] * self.hyperparam.theta_f
            K2[i, n-1] = K2[i, n-1] * self.hyperparam.theta_f

        K2[n-1, n-1] = K2[n-1, n-1] + self.hyperparam.theta_n * self.hyperparam.theta_n
        K2[n-1, 0:n-1] = (K2[0:n-1, n-1]).T

        # print('k_matrix', self.k_matrix)
        # print('new k_matrix', K2)
        # print('inv_k_matrix', self.inv_k_matrix)
        e = self.inv_k_matrix[0][0]
        # print('e', e)
        f = self.inv_k_matrix[1:n, 0].reshape((n-1, 1))
        # print('f', f)
        g = K2[n-1, n-1]
        # print('g', g)
        h = K2[0:n-1, n-1].reshape((n-1, 1))
        # print('h', h)
        H = self.inv_k_matrix[1:n, 1:n]
        # print('H', H)
        B = H - (f.dot(f.T)) / e
        # print('B', B)
        s = 1 / (g - (h.T).dot(B.dot(h)))
        # print('s', s)
        haha_11 = B + (B.dot(h)).dot((B.dot(h)).T) * s
        haha_12 = -B.dot(h) * s
        haha_21 = -(B.dot(h)).T * s
        haha_22 = s
        temp_1 = np.hstack((haha_11, haha_12))
        temp_2 = np.hstack((haha_21, haha_22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))


        # update delta
        self.delta.popleft()
        self.delta.append(0)
        for i in range(n-1):
            self.delta[i] = self.delta[i] - self.k_matrix[0, i+1]

        for i in range(n-1):
            self.delta[i] = self.delta[i] + K2[n-1, i]

        for i in range(n):
            self.delta[n-1] = self.delta[n-1] + K2[i, n-1]

        self.k_matrix = K2

    def SM_update_SE_kernel(self, new_x, new_y, index):
        n = len(self.kernel_x)
        self.kernel_x[index] = new_x
        self.kernel_y[index] = new_y
        new_k_matrix = self.k_matrix.copy()
        for i in range(n):
            new_k_matrix[i, index] = np.sum(np.square(self.kernel_x[i] - self.kernel_x[index]))
            new_k_matrix[i, index] = new_k_matrix[i, index] / -2
            new_k_matrix[i, index] = new_k_matrix[i, index] / self.hyperparam.len
            new_k_matrix[i, index] = new_k_matrix[i, index] / self.hyperparam.len
            new_k_matrix[i, index] = np.exp(new_k_matrix[i, index])
            new_k_matrix[i, index] = new_k_matrix[i, index] * self.hyperparam.theta_f
            new_k_matrix[i, index] = new_k_matrix[i, index] * self.hyperparam.theta_f

        new_k_matrix[index, index] = new_k_matrix[index, index] + self.hyperparam.theta_n * self.hyperparam.theta_n
        for i in range(n):
            new_k_matrix[index, i] = new_k_matrix[i, index]

        r = new_k_matrix[:, index].reshape((n, 1)) - self.k_matrix[:, index].reshape((n, 1))
        A = self.inv_k_matrix - (self.inv_k_matrix.dot(r.dot(self.inv_k_matrix[index, :].reshape((1, n)))))/(1 + r.transpose().dot(self.inv_k_matrix[:, index].reshape((n, 1)))[0, 0])
        self.inv_k_matrix = A - ((A[:, index].reshape((n, 1))).dot(r.transpose().dot(A)))/(1 + (r.transpose().dot(A[:, index].reshape((n, 1))))[0, 0])

        # update delta

        for i in range(n):
            if i!=index:
                self.delta[i] = self.delta[i] - self.k_matrix[index, i]

        for i in range(n):
            if i != index:
                self.delta[i] = self.delta[i] + new_k_matrix[index, i]

        self.delta[index] = 0
        for i in range(n):
            self.delta[index] = self.delta[index] + new_k_matrix[i, index]

        self.k_matrix = new_k_matrix

    def count_delta(self, new_x):
        n = len(self.kernel_x)
        temp_delta = np.zeros((1, n))
        for i in range(n):
            temp_delta[0, i]= np.sum(np.square(self.kernel_x[i] - new_x))
            temp_delta[0, i] = temp_delta[0, i] / -2
            temp_delta[0, i] = temp_delta[0, i] / self.hyperparam.len
            temp_delta[0, i] = temp_delta[0, i] / self.hyperparam.len
            temp_delta[0, i] = np.exp(temp_delta[0, i])
            temp_delta[0, i] = temp_delta[0, i] * self.hyperparam.theta_f
            temp_delta[0, i] = temp_delta[0, i] * self.hyperparam.theta_f
        temp_delta = np.sum(temp_delta)
        return temp_delta

    def get_max(self, delta):
        max_index = 0
        max_value = delta[0]
        for i in range(1, len(delta)):
            if delta[i] > max_index:
                max_index = i
                max_value = delta[i]
        return max_value, max_index
    

###################################################
######        batch incremental GPR         #######
###################################################

def matrix_block_inversion(Ainv,B,C,D):
    assert Ainv.shape[0] == Ainv.shape[1]
    assert D.shape[0] == D.shape[1]
    assert B.shape[0] == Ainv.shape[0] and B.shape[1] == D.shape[0]
    assert C.shape[0] == D.shape[0] and C.shape[1] == Ainv.shape[0]

    S = D - C @ Ainv @ B
    Sinv = np.linalg.inv(S)

    Xinv00 = Ainv + Ainv @ B @ Sinv @ C @ Ainv
    Xinv01 = -Ainv @ B @ Sinv
    Xinv10 = -Sinv @ C @ Ainv
    Xinv11 = Sinv

    Xinv = np.block([   [Xinv00, Xinv01],
                        [Xinv10, Xinv11]   ])

    return Xinv

def matrix_inverse_remove_indices(A, i_s):
    # check A square matrix
    n = A.shape[0]
    assert A.shape[1] == n, "Ainv must be a square matrix"

    # order i_s in descending order
    assert len(i_s) > 0, "i_s must be a non-empty list"
    i_s = np.sort(np.array(i_s))[::-1]

    # call x_s all As indices
    x_s = np.arange(A.shape[0])
    adj_v = np.zeros(len(x_s), np.int32)

    swap = -1
    for i in i_s:
        x_s[[i, swap]] = x_s[[swap, i]]
        swap -= 1

    A[:, :] = A[x_s, :]
    A[:, :] = A[:, x_s]

    amt = i_s.shape[0]
    a = A[:-amt, :-amt]
    b = A[:-amt, -amt:]
    c = A[-amt:, :-amt]
    d = A[-amt:, -amt:]

    dinv = np.linalg.inv(d)

    result = a - np.matmul(b, np.matmul(dinv, c))

    for i in i_s:
        adj_v[x_s > i] += 1

    x_s-=adj_v
    x_s = x_s[:-amt]

    result[x_s,:] = result[np.arange(result.shape[0]), :]
    result[:,x_s] = result[:, np.arange(result.shape[0])]

    return result

class BIGPR(object):
    '''
    This project implements the Incremental Gaussian Process Regression (IGPR) algorithm adding support for batch learning.
    Training a model via learning one sample at a time is also supported
    All the code involved has been mostly rewritten to improve performance and optimization,
    exploiting the vectorial capabilities of numpy and the use of matrix inversion lemmas.

    Learning batches and learning single samples might not provide the same k_matrix
    '''
    def __init__(self, init_x, init_y, max_k_matrix_size=400, hyperparam=HyperParam(1, 1, 0.1)):
        self.hyperparam = hyperparam
        self.max_k_matrix_size = max_k_matrix_size
        self.lamda = 1
        self.count = 0
        self.kernel_x = deque()
        self.kernel_y = deque()
        self.kernel_x.append(init_x)
        self.kernel_y.append(init_y)
        self.k_matrix = np.ones((1, 1)) + self.hyperparam.theta_n * self.hyperparam.theta_n
        self.inv_k_matrix = np.ones((1, 1)) / (self.hyperparam.theta_n * self.hyperparam.theta_n)
        self.is_av = False
        temp = np.sum(self.k_matrix, axis=0)
        self.delta = deque()
        for i in range(temp.shape[0]):
            self.delta.append(temp[i])
        
        # self.informativity = deque().append(0.)        # i.e. max covariance wrt other samples
        self.info_mat = deque().append(0.)             # i.e. covariance wrt other samples, ordered, excluding self

        self.samples_substituted_count = 0
        self.samples_substituted = []

    def is_available(self):
        if not self.is_av:
            self.is_av = len(self.kernel_x) >= 2
        return self.is_av

    def learn(self, new_x, new_y):
        self.delta = deque(np.array(self.delta) * self.lamda)

        if not self.is_available():
            self.kernel_x.append(new_x)
            self.kernel_y.append(new_y)
            self.calculate_SE_kernel()
            self.inv_k_matrix = np.linalg.inv(self.k_matrix)

        elif len(self.kernel_x) < self.max_k_matrix_size:
            self.aug_update_SE_kernel(new_x, new_y)

        else:
            # call the same as the batch method because more optimized and gives same result
            self.aug_update_SE_kernel(new_x, new_y)
            self.remove_kernel_samples(1)
            # OLD WAY -> slightly less efficient
            # self.sub_kernel_sample(new_x, new_y)
            
    def learn_batch(self, new_xs, new_ys):
        self.delta = deque(np.array(self.delta) * self.lamda)

        if not self.is_available():
            self.kernel_x.extend(new_xs)
            self.kernel_y.extend(new_ys)
            self.calculate_SE_kernel()
            self.inv_k_matrix = np.linalg.inv(self.k_matrix)
            #check if max size overshoot 
            if len(self.kernel_x) > self.max_k_matrix_size:
                self.remove_kernel_samples( len(self.kernel_x) - self.max_k_matrix_size )

        # contained in last elif
        # elif len(self.kernel_x) == self.max_k_matrix_size:
        #     # if matrix already maxxed, update for amt=new_xs.shape[0]
        #     self.batch_aug_update_SE_kernel(new_xs, new_ys)
        #     self.remove_kernel_samples(new_xs.shape[0])

        elif len(self.kernel_x) + len(new_xs) < self.max_k_matrix_size:
            # if matrix + new_xs not maxxed yet, insert new samples
            self.batch_aug_update_SE_kernel(new_xs, new_ys)

        elif len(self.kernel_x) + len(new_xs) >= self.max_k_matrix_size:
            new_xs, new_ys, _, _ = self.screen_new_samples(new_xs, new_ys)
            if len(new_xs) == 0:
                return

            # otherwise, if matrix not maxxed, but will be after adding new_xs
            # add new_xs but only remove a smaller amt
            # using this formula:
            amt_toremove = len(new_xs) + len(self.kernel_x) - self.max_k_matrix_size

            #now we add all and remove to come back to the max size
            self.batch_aug_update_SE_kernel(new_xs, new_ys)
            self.remove_kernel_samples(amt_toremove)

        else:
            print("ERROR: shouldn't be here")
            exit(1)

    def calculate_SE_kernel(self, kernel_x=None, return_values=False):
        if kernel_x is None:
            kernel_x = self.kernel_x
        
        n = len(kernel_x)

        #compute kernel matrix
        k_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                k_matrix[i][j] = np.sum(np.square(kernel_x[i] - kernel_x[j]))
                k_matrix[i][j] = k_matrix[i][j] / (-2)
                k_matrix[i][j] = k_matrix[i][j] / self.hyperparam.len
                k_matrix[i][j] = k_matrix[i][j] / self.hyperparam.len
                k_matrix[i][j] = np.exp(k_matrix[i][j])
                k_matrix[i][j] = k_matrix[i][j] * self.hyperparam.theta_f
                k_matrix[i][j] = k_matrix[i][j] * self.hyperparam.theta_f
        k_matrix = k_matrix + self.hyperparam.theta_n * self.hyperparam.theta_n * np.eye(n)
        infomat = self.compute_info_mat(k_matrix)

        #compute delta
        d = np.sum(k_matrix, axis=0)
        delta = deque(d)

        if return_values:
            # return k_matrix, delta, info
            return k_matrix, delta, infomat
        else:
            self.k_matrix = k_matrix
            self.info_mat = infomat
            self.delta = delta
            # self.informativity = informativity

    def compute_info_mat(self, kmat):
        infomat = deque()
        for i in range(kmat.shape[0]):
            temp = kmat[i,:].copy()
            temp[i] = 0
            infomat.append(np.argsort(temp)[::-1])
        
        return infomat

    def predict(self, coming_xs):
        if coming_xs.ndim == 1:
            coming_xs = coming_xs.reshape((1, -1))
        predictions = deque()
        for coming_x in coming_xs:
            if self.is_available():
                k_x = np.array(self.kernel_x, dtype=np.float32)
                cross_kernel_k = np.sum(np.square(k_x - coming_x), axis=1)
                cross_kernel_k /= -2 * self.hyperparam.len * self.hyperparam.len
                cross_kernel_k = np.exp(cross_kernel_k)
                cross_kernel_k *= self.hyperparam.theta_f * self.hyperparam.theta_f
                cross_kernel_k = cross_kernel_k.reshape((1, -1))

                kernel_y_mat = np.array(self.kernel_y)
                prediction = cross_kernel_k.dot(self.inv_k_matrix.dot(kernel_y_mat))
                predictions.append(prediction.flatten())
            else:
                prediction = self.kernel_y[0]
                predictions.append(prediction)
        return np.array(predictions).flatten()

    def aug_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)
        self.kernel_x.append(new_x)
        self.kernel_y.append(new_y)
        k_x = np.array(self.kernel_x)

        new_row = np.sum(np.square(k_x - new_x) , axis=1) 
        new_row /= (-2 * self.hyperparam.len * self.hyperparam.len)
        new_row = np.exp(new_row)
        new_row *= self.hyperparam.theta_f * self.hyperparam.theta_f
        new_row = new_row.reshape(1, -1)

        self.k_matrix = np.vstack((self.k_matrix, new_row[:,:-1]))
        self.k_matrix = np.hstack((self.k_matrix, new_row.T))

        self.k_matrix[n, n] += self.hyperparam.theta_n * self.hyperparam.theta_n

        # NB using block inversion is worse, 2.5x time

        # compute inv matrix 
        b = self.k_matrix[0:n, n].reshape((n, 1))
        d = self.k_matrix[n, n]
        e = self.inv_k_matrix.dot(b)
        g = 1 / (d - (b.T).dot(e))
        haha_11 = self.inv_k_matrix + g[0][0]*e.dot(e.T)
        haha_12 = -g[0][0]*e
        haha_21 = -g[0][0]*(e.T)
        haha_22 = g
        temp_1 = np.hstack((haha_11, haha_12))
        temp_2 = np.hstack((haha_21, haha_22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))

        
        # udpate delta        
        d += self.k_matrix[:-1, n]
        d = np.append(d, 0)
        d[n] += self.k_matrix[:, n].sum()
        self.delta = deque(d)

        # WHEN ADDING IN THE NOT-BATCHED, I NEED TO PUT TO 0 THE SELF-COVARIANCE TERM!
        row = new_row[0].copy()
        row[n] = 0
        new_info_row = np.argsort(row)[::-1]
        self.info_mat.append(new_info_row)

    def screen_new_samples(self, new_xs, new_ys):
        xs = new_xs.copy()
        ys = new_ys.copy()
        k_x = np.array(self.kernel_x)

        # calculate new submatrix (i.e. new_xs kernel)
        new_k_matrix, _, new_infomat = self.calculate_SE_kernel(kernel_x=xs, return_values=True)

        new_infomat_maxidx = np.array([new_infomat[i][0] for i in range(len(new_infomat))])
        new_infomat_maxval = new_k_matrix[np.arange(len(new_infomat)), new_infomat_maxidx]

        killed_idxs = []

        while xs.shape[0] > 1 and new_infomat_maxval.max() >= 0.95: #dont even bother adding if informativity is too high (0.95 but it could be even 0.8)
            kill = np.sort( new_infomat_maxidx[np.where(new_infomat_maxval == new_infomat_maxval.max())] )[::-1][0]
            killed_idxs.append(kill)
            new_k_matrix = np.delete(new_k_matrix, kill, axis=0)
            new_k_matrix = np.delete(new_k_matrix, kill, axis=1)
            new_infomat = self.compute_info_mat(new_k_matrix)
            xs = np.delete(xs, kill, axis=0)
            ys = np.delete(ys, kill, axis=0)

            new_infomat_maxidx = np.array([new_infomat[i][0] for i in range(len(new_infomat))])
            new_infomat_maxval = new_k_matrix[np.arange(len(new_infomat)), new_infomat_maxidx]

        # print worst self.info_mat values, in amount as xs.shape[0]
        self_infomat_maxidx = np.array([self.info_mat[i][0] for i in range(len(self.info_mat))])
        self_infomat_maxval = self.k_matrix[np.arange(len(self.info_mat)), self_infomat_maxidx]
        self_killable_val = np.sort(self_infomat_maxval)[:xs.shape[0]]
        # self_killable_idx = self_infomat_maxidx[np.argsort(self_infomat_maxval)[:xs.shape[0]]]

        # calculate rectangular matrix -> informativity
        xs_info = []
        for i,x in enumerate(xs):
            row = np.sum(np.square(k_x - x), axis=1) / (-2 * self.hyperparam.len * self.hyperparam.len)
            row = np.exp(row) * self.hyperparam.theta_f * self.hyperparam.theta_f
            worst_i = np.argsort(row)[::-1][0]
            xs_info.append(row[worst_i])

        dont_add = []

        for i in np.argsort(xs_info):
            if xs_info[i] > self_killable_val[-1]:
                dont_add.append(i)
            else:
                self_killable_val[-1] = xs_info[i]
                self_killable_val = np.sort(self_killable_val)

        xs = np.delete(xs, dont_add, axis=0)
        ys = np.delete(ys, dont_add, axis=0)

        return xs, ys, killed_idxs, dont_add

    def batch_aug_update_SE_kernel(self, new_xs, new_ys):
        k_x = np.array(self.kernel_x)
        self.kernel_x.extend(new_xs)
        self.kernel_y.extend(new_ys)

        # calculate new submatrix (i.e. new_xs kernel)
        new_k_matrix, _, _ = self.calculate_SE_kernel(kernel_x=new_xs, return_values=True)
        
        # calculate rectangular matrix
        new_rows = np.array([np.sum(np.square(k_x - new_x), axis=1) for new_x in new_xs])
        new_rows /= (-2 * self.hyperparam.len * self.hyperparam.len)
        new_rows = np.exp(new_rows.astype(np.float32))
        new_rows *= self.hyperparam.theta_f * self.hyperparam.theta_f

        self.inv_k_matrix = matrix_block_inversion( Ainv=self.inv_k_matrix, B=new_rows.T, C=new_rows, D=new_k_matrix )

        # compose the new overall matrix
        self.k_matrix = np.vstack(( self.k_matrix, new_rows))
        self.k_matrix = np.hstack(( self.k_matrix, np.vstack((new_rows.T, new_k_matrix))    ))

        # assert np.allclose(self.inv_k_matrix, np.linalg.inv(self.k_matrix)), "Inverse matrix is not correct"

        # update delta
        d = np.array(self.delta)
        d += new_rows.sum(axis=0)
        d_new = np.vstack((new_rows.T, new_k_matrix)).sum(axis=0)
        d = np.append(d, d_new)
        self.delta = deque(d)

        # update info mat
        rows = self.k_matrix[-len(new_xs):, :].copy()
        new_info_rows=[]
        for i in range(len(new_xs)):
            rows[i, -len(new_xs)+i] = 0.
            new_info_rows.append(np.argsort(rows[i])[::-1])
        self.info_mat.extend(np.array(new_info_rows))

    def sub_kernel_sample(self, new_x, new_y):
        # new_delta = self.count_delta(new_x) # not using it
        # max_value, max_index = self.get_max(self.delta)
        new_info = self.get_sample_informativity(self.kernel_x, new_x)
        info = np.array([self.info_mat[i][0] for i in range(len(self.info_mat))])
        info_values = self.k_matrix[np.arange(len(info)), info]
        max_value, max_index = np.max(info_values), np.argmax(info_values)
        pass
        if new_info < max_value:
            self.samples_substituted_count += 1
            self.samples_substituted.append(max_index)

            # self.schur_update_SE_kernel(new_x, new_y)
            self.SM_update_SE_kernel(new_x, new_y, max_index)
            self.count = self.count + 1
            if self.count > 0:#int(self.max_k_matrix_size/3):
                self.count = 0
                self.calculate_SE_kernel()
                self.inv_k_matrix = np.linalg.inv(self.k_matrix)

    def schur_update_SE_kernel(self, new_x, new_y):
        n = len(self.kernel_x)

        self.kernel_x.append(new_x)
        self.kernel_y.append(new_y)
        self.kernel_x.popleft()
        self.kernel_y.popleft()

        K2 = self.k_matrix[1:n, 1:n]

        K2[:, -1] = np.sum(np.square(self.kernel_x - new_x), axis=1)
        K2[:, -1] /= (-2 * self.hyperparam.len * self.hyperparam.len)
        K2[:, -1] = np.exp(K2[:, -1])
        K2[:, -1] *= self.hyperparam.theta_f * self.hyperparam.theta_f

        K2[n-1, n-1] += self.hyperparam.theta_n * self.hyperparam.theta_n
        K2[n-1, 0:n-1] = (K2[0:n-1, n-1]).T        

        # print('k_matrix', self.k_matrix)
        # print('new k_matrix', K2)
        # print('inv_k_matrix', self.inv_k_matrix)
        e = self.inv_k_matrix[0][0]
        # print('e', e)
        f = self.inv_k_matrix[1:n, 0].reshape((n-1, 1))
        # print('f', f)
        g = K2[n-1, n-1]
        # print('g', g)
        h = K2[0:n-1, n-1].reshape((n-1, 1))
        # print('h', h)
        H = self.inv_k_matrix[1:n, 1:n]
        # print('H', H)
        B = H - (f.dot(f.T)) / e
        # print('B', B)
        s = 1 / (g - (h.T).dot(B.dot(h)))
        # print('s', s)
        haha_11 = B + (B.dot(h)).dot((B.dot(h)).T) * s
        haha_12 = -B.dot(h) * s
        haha_21 = -(B.dot(h)).T * s
        haha_22 = s
        temp_1 = np.hstack((haha_11, haha_12))
        temp_2 = np.hstack((haha_21, haha_22))
        self.inv_k_matrix = np.vstack((temp_1, temp_2))

        # update delta
        self.delta.popleft()
        self.delta.append(0)

        self.delta -= self.k_matrix[0, :-1]
        self.delta += K2[n-1, :n-1]
        self.delta[n-1] += K2[0:n-1, n-1].sum()

        # update info mat -> upon removal, compute from scratch
        self.info_mat = self.compute_info_mat(self.k_matrix)

        # update k_matrix
        self.k_matrix = K2

    def SM_update_SE_kernel(self, new_x, new_y, index):
        n = len(self.kernel_x)
        self.kernel_x[index] = new_x
        self.kernel_y[index] = new_y
        new_k_matrix = self.k_matrix.copy()

        new_k_matrix[:, index] = np.sum(np.square(self.kernel_x - self.kernel_x[index]), axis=1)
        new_k_matrix[:, index] /= (-2 * self.hyperparam.len * self.hyperparam.len)
        new_k_matrix[:, index] = np.exp(new_k_matrix[:, index])
        new_k_matrix[:, index] *= self.hyperparam.theta_f * self.hyperparam.theta_f

        new_k_matrix[index, index] += self.hyperparam.theta_n * self.hyperparam.theta_n
        new_k_matrix[index, :] = (new_k_matrix[:, index]).T

        r = new_k_matrix[:, index].reshape((n, 1)) - self.k_matrix[:, index].reshape((n, 1))
        A = self.inv_k_matrix - (self.inv_k_matrix.dot(r.dot(self.inv_k_matrix[index, :].reshape((1, n)))))/(1 + r.transpose().dot(self.inv_k_matrix[:, index].reshape((n, 1)))[0, 0])
        self.inv_k_matrix = A - ((A[:, index].reshape((n, 1))).dot(r.transpose().dot(A)))/(1 + (r.transpose().dot(A[:, index].reshape((n, 1))))[0, 0])

        # update delta
        d = np.array(self.delta)
        d -= self.k_matrix[index, :]
        d += new_k_matrix[index, :]
        d[index] = np.sum(new_k_matrix[:, index])
        self.delta = deque(d)

        # update info mat -> upon removal, compute from scratch
        self.info_mat = self.compute_info_mat(new_k_matrix)

        self.k_matrix = new_k_matrix

    def remove_kernel_samples(self, amount):
        '''
        removes an amount of samples from the kernel,
        so to make the kernel size back to its maximum size when exceeding

        after some tries, the most effective way to do this is to:
        - choose all the samples with max correlation with another sample
        - among those, choose the ones with a highest delta (sum of all correlations)
        - if more than one, remove the highest index (keep oldest, discard newest, to avoid exploring forever)

        for i in I:
            remove:
                - self.kernel_x[i]
                - self.kernel_y[i]
                - self.k_matrix[i, :]
                - self.k_matrix[:, i]
                - self.inv_k_matrix[i, :]   -> i cannot just do this
                - self.inv_k_matrix[:, i]
                - self.delta[i]
        '''
        if amount<=0:
            return

        infomat = np.array(self.info_mat, dtype=object)
        infomat = list(infomat)
        kmat = np.array(self.k_matrix)
        delta = np.array(self.delta)

        kill_list = []

        for _ in range(amount):
            info_max_idxs = np.array([inforow[0] for inforow in infomat])        #infomat already sorted
            info_max_vals = kmat[np.arange(len(info_max_idxs)), info_max_idxs]
            
            # filter by max correlation
            max_correlation_idxs = np.argwhere(info_max_vals == np.amax(info_max_vals)).flatten()
            max_correlation_idxs = np.unique((info_max_idxs[max_correlation_idxs], max_correlation_idxs))    # info[argmaxs] are the simmetric indices (not always present in argmaxs since infomat is "diagonal", deque of arrays)

            # filter by worse delta
            max_delta_idxs = np.argwhere(delta[max_correlation_idxs] == np.amax(delta[max_correlation_idxs])).flatten()

            # filter by oldest (highest index)
            kill = np.sort(max_correlation_idxs[max_delta_idxs].flatten())[-1]      # if more than one, remove newest
            kill_list.append(kill)

            # UPDATE infomat, delta, kmat
            #remove `kill` from  infomat , recompute  info  at the beginning of the loop
            infomat[kill] = np.zeros(len(infomat[kill]), dtype=np.int32)
            for i in range(len(infomat)): 
                infomat[i]=np.delete(infomat[i], np.where(np.isin(infomat[i], kill)))
                # infomat[i][infomat[i] > kill]-=1        # adjust indices numbers
            delta -= kmat[kill, :]
            # delta = np.delete(delta, kill)
            # kmat = np.delete(kmat, kill, axis=0)
            # kmat = np.delete(kmat, kill, axis=1)


        #kill stuff
        kill_list = np.sort(kill_list)[::-1]

        kmat = np.delete(kmat, kill_list, axis=0)
        kmat = np.delete(kmat, kill_list, axis=1)
        infomat = np.array(infomat)
        infomat = np.delete(infomat, kill_list, axis=0)
        delta = np.delete(delta, kill_list)
        self.inv_k_matrix = matrix_inverse_remove_indices(self.inv_k_matrix, kill_list)
        
        for kill in kill_list:
            del self.kernel_x[kill]
            del self.kernel_y[kill]
            for i in range(len(infomat)): 
                # if i delete idx 1000, idx 1001 will become 1000, etc...
                infomat[i][infomat[i] > kill]-=1        # adjust indices numbers

        self.samples_substituted_count += len(kill_list)
        self.samples_substituted.append(kill_list)

        # update info mat -> upon removal, compute from scratch
        self.k_matrix = kmat
        # self.info_mat = self.compute_info_mat(kmat)
        # infomat doesn't need to be recomputed
        self.info_mat = deque(infomat)
        self.delta = deque(delta)

        # assert np.allclose( np.abs(np.rint(np.matmul(self.k_matrix, self.inv_k_matrix))), np.eye(len(self.k_matrix)) )

        assert len(self.kernel_x) == len(self.kernel_y) == self.k_matrix.shape[0] == self.k_matrix.shape[1] == self.inv_k_matrix.shape[0] == self.inv_k_matrix.shape[1] == len(self.delta) == len(self.info_mat) == self.max_k_matrix_size,\
            "not all kernel structures have the same size after removal"

    def count_delta(self, new_x):
        '''
        this functions provides a metric to choose the informativity of a sample
        but imagine a sample S is the same to an already existing sample,
        that is, though, very informative, hence low correlation with all the other samples

        S would be added for sure, since its delta is very low, but it is not informative at all,
        since it is a duplicate

        i am rewriting to use as a metric the single highest correlation with the other samples
        '''
        n = len(self.kernel_x)

        d = np.sum(np.square(self.kernel_x - new_x), axis=1)
        d /= (-2 * self.hyperparam.len * self.hyperparam.len)
        d = np.exp(d)
        d *= self.hyperparam.theta_f * self.hyperparam.theta_f
        d = np.sum(d)

        return d

    def get_sample_informativity(self, kernel_x, x):
        '''
        This functions provides a metric to choose the informativity of a sample:
        returns the highest correlation between the new sample and the already existing samples.

        If x is present in kernel_x, be sure to pass kernel_x removing it first,
        otherwise this function will return 1.0
        '''
        d = np.sum(np.square(kernel_x - x), axis=1)    #still summing over the features dimension
        d /= (-2 * self.hyperparam.len * self.hyperparam.len)
        d = np.exp(d)
        d *= self.hyperparam.theta_f * self.hyperparam.theta_f
        return np.max(d)

    def calculate_array_sample_distances(self, kernel_x, new_x):
        '''
        This function takes in input a list of kernel samples,
        calulates the distance with new_x for each sample,
        and returns the np array of the distances.
        '''
        d = np.sum(np.square(kernel_x - new_x), axis=1)
        return d



###########################################
######        CVR Estimation        #######
###########################################

'''
in CVR estimation
X are contexts of users
y are boolean values for converted/not converted
'''
# regression model can be SGDRegression, IGPR, GaussianProcessRegressor
# IGPR has different update and initialization
# GPR has different update
class CVR_Estimator(object):
    def __init__(self, regression_model):
        self.regression_model = regression_model
        self.regressor = eval(regression_model)()
        pass

    def update(self, x, y):
        self.regressor.learn(x, y)
        pass

    def predict(self, x):
        return self.regressor.predict(x)