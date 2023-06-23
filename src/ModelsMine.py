#################################################################
##### https://github.com/pranz24/pytorch-soft-actor-critic  #####
#################################################################
# TODO: citare la repo

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

    def load_csv(self, file_name):
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            columns = [row for row in reader]
        columns = np.array(columns)
        m_x, n_x = columns.shape
        data_set = np.zeros((m_x,n_x))
        for i in range(m_x):
            for j in range(n_x):
                data_set[i][j] = float(columns[i][j])
        return data_set

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