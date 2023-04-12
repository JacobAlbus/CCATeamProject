from bandit import Bandit

import os
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

FEEDBACK_DIM = 20
ACTION_DIM = 21

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        
        self.hidden_size = 128
        self.double()

        self.input_layer = nn.Linear(FEEDBACK_DIM + 1, self.hidden_size)  # + 1 for the context
        self.relu1 = nn.ReLU()
        self.hidden_layer1 = nn.Linear(self.hidden_size, ACTION_DIM)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        output = self.input_layer(state)
        output = self.relu1(output)
        output = self.hidden_layer1(output)
        return self.softmax(output)

class VFA(nn.Module):
    def __init__(self):
        super(VFA, self).__init__()
        
        self.hidden_size = 128
        self.double()

        self.input_layer = nn.Linear(FEEDBACK_DIM + 1, self.hidden_size) # + 1 for the context
        self.relu1 = nn.ReLU()
        self.hidden_layer1 = nn.Linear(self.hidden_size, 1)

    def forward(self, state):
        output = self.input_layer(state)
        output = self.relu1(output)
        return self.hidden_layer1(output)

def select_action(state, vfa_model, policy_model):
    action_prob_dist = policy_model(state)
    state_value = vfa_model(state)

    # The current problem is that after training for only 1 minute,
    # the policy model starts producing probability distributions with Nan values
    # or with 100% probability for a wrong option. Perhaps there's a problem with the
    # bandit environment code.
    try:
        N = int((ACTION_DIM - 1) / 2)
        actions = np.array([num for num in range(-N, N+1)])
        action = np.random.choice(actions, p=action_prob_dist.data.numpy())
    except ValueError:
        print(action_prob_dist, state)
        exit("NAN PROBABLITIY") 

    action_prob = action_prob_dist[action]
    
    return action, torch.log(action_prob), state_value

def calculate_loss(rewards, saved_steps, gamma=0.999):
    R = 0
    returns = []
    eps = np.finfo(np.float32).eps.item()

    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)

    if len(rewards) == 1:
        # Because we only sample a single episode, I manually calculated the true mean and std of the returns
        returns_mean = -16.5 
        returns_std = 19.438 
        returns = (returns - returns_mean) / (returns_std + eps)
    else:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    
    state_values = torch.stack([values for (log_prob, values) in saved_steps]).squeeze()
    action_probs = torch.stack([log_prob for (log_prob, values) in saved_steps])

    estimator = returns

    policy_losses = -torch.sum(action_probs * estimator)
    value_losses = torch.sum((state_values - estimator)**2)
    return value_losses, policy_losses

def train(vfa_model, policy_model, lr=0.01):
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=lr)
    policy_scheduler = Scheduler.StepLR(policy_optimizer, step_size=100, gamma=0.99)    

    vfa_optimizer = optim.Adam(vfa_model.parameters(), lr=lr)
    vfa_scheduler = Scheduler.StepLR(vfa_optimizer, step_size=100, gamma=0.99)    

    ewma_reward = 0

    for i_episode in count(1):
        curr_state = env.init(feedback_type="no_hindsight")
        ep_reward = 0

        rewards = []
        saved_steps = []
        for _ in range(100):
            action, log_action_prob, state_value = select_action(curr_state, vfa_model, policy_model)
            curr_state, reward, done = env.step(action, feedback_type="no_hindsight")
            ep_reward += reward

            rewards.append(reward)
            saved_steps.append((log_action_prob, state_value))
        
        vfa_loss, policy_loss = calculate_loss(rewards, saved_steps)
        saved_steps = []

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        policy_scheduler.step()

        vfa_optimizer.zero_grad()
        vfa_loss.backward()
        vfa_optimizer.step()
        vfa_scheduler.step()
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        if i_episode % 100 == 0:
            print('Episode {} \treward: {}\t ewma reward: {}'.format(i_episode, ep_reward, ewma_reward))

if __name__ == '__main__':
    random_seed = 12
    lr = 0.0001

    N = int((ACTION_DIM - 1) / 2)
    env = Bandit(K=FEEDBACK_DIM, N=N, variance=10)
    torch.manual_seed(random_seed)  

    vfa_model = VFA()
    policy_model = Policy()
    train(vfa_model, policy_model, lr)

