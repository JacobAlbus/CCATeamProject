from collections import namedtuple
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from bandit import Bandit

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
ewma_reward_threshold = -1

class Policy(nn.Module):
    def __init__(self, observation_dim, hidden_size, action_dim):
        super(Policy, self).__init__()
        self.double()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        action_prob = self.net(state)
        return action_prob

class VFA(nn.Module):
    def __init__(self, observation_dim, hidden_size):
        super(VFA, self).__init__()
        self.double()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
    )
    def forward(self, state):
        state_value = self.net(state)
        return state_value

class ModelHolder:

    def __init__(self, env, lr):
        # Extract the dimensionality of state and action spaces
        self.discrete = True
        self.observation_dim = 1
        self.action_dim = env.ACTION_DIM
        self.hidden_size = 32

        # construct policy and vfa network
        self.policy_model = Policy(self.observation_dim, self.hidden_size, self.action_dim)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)

        self.vfa_model = VFA(self.observation_dim, self.hidden_size)
        self.vfa_optimizer = optim.Adam(self.vfa_model.parameters(), lr=lr)

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def select_action(self, state):
        X = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action_prob = self.policy_model(X)
        state_value = self.vfa_model(X)
        m = Categorical(action_prob)
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    # calculate losses and backpropagate through networks
    def update(self):
        self.policy_optimizer.zero_grad()
        self.vfa_optimizer.zero_grad()
        policy_loss, vfa_loss = self.calculate_loss()
        policy_loss.backward()
        vfa_loss.backward()
        self.policy_optimizer.step()
        self.vfa_optimizer.step()
        self.clear_memory()

    def calculate_loss(self, gamma=0.99):

        # Initialize the lists and variables
        Gt = 0.0
        saved_actions = self.saved_actions
        policy_losses = []
        value_losses = []

        for t in reversed(range(len(self.rewards))):
            Gt = Gt * gamma + self.rewards[t]
            log_prob, state_value = saved_actions[t]
            advantage = Gt - state_value.detach().item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append((state_value[0] - Gt)**2)  # simple MSE loss

        policy_loss = torch.stack(policy_losses).sum()
        vfa_loss = torch.stack(value_losses).sum()

        return policy_loss, vfa_loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

def train(env, lr=0.01):

    # Instantiate the models
    model_holder = ModelHolder(env, lr)

    # EWMA (Exponential Weighted Moving Average) reward for tracking the learning progress
    ewma_reward = 0
    ewma_move = 0.005

    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        done = False

        # run episode
        while done is False:
            t += 1
            action = model_holder.select_action(state)
            (feedback, state), reward, done = env.step(action)
            model_holder.rewards.append(reward)
            ep_reward += reward

        # update policy
        model_holder.update()

        # update EWMA reward and log the results
        ewma_reward = ewma_move * ep_reward + (1 - ewma_move) * ewma_reward
        if i_episode % 1000 == 0:
            print('Episode {}\treward: {}\t ewma reward: {}'.format(i_episode, ep_reward, ewma_reward))

        if i_episode > 10000 and ewma_reward > ewma_reward_threshold:
            print("Solved in Episode {}! Running reward is now {}!".format(i_episode, ewma_reward))
            break


def main():
    # For reproducibility, fix the random seed
    random_seed = 1
    lr = 0.0004
    env = Bandit(K=3, N=10, variance=0.1, seed=random_seed)
    torch.manual_seed(random_seed)
    train(env, lr)

if __name__ == '__main__':
    main()