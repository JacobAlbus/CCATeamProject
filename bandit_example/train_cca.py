from bandit_example.bandit import Bandit
from gru import GRUNet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as Scheduler

# class Policy(nn.Module):
#     """
#         Implement both policy network and the value network in one model
#         - Note that here we let the actor and value networks share the first layer
#         - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
#         - Feel free to add any member variables/functions whenever needed
#         TODO:
#             1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
#             2. Random weight initialization of each layer
#     """
#     def __init__(self):
#         super(Policy, self).__init__()
        
#         # Extract the dimensionality of state and action spaces
#         self.hidden_size = 128
#         self.double()
        
#         ########## YOUR CODE HERE (5~10 lines) ##########
#         self.layer1 = nn.Linear(self.observation_dim, self.hidden_size)
#         self.relu1 = nn.ReLU()

#         self.action_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.action_relu2 = nn.ReLU()
#         self.action_layer3 = nn.Linear(self.hidden_size, self.action_dim)
#         self.action_softmax = nn.Softmax(dim=0)

#         self.value_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
#         self.value_relu2 = nn.ReLU()
#         self.value_layer3 = nn.Linear(self.hidden_size, 1)
#         ########## END OF YOUR CODE ##########
        
#         # action & reward memory
#         self.saved_actions = []
#         self.rewards = []

#     def forward(self, state):
#         """
#             Forward pass of both policy and value networks
#             - The input is the state, and the outputs are the corresponding 
#               action probability distirbution and the state value
#             TODO:
#                 1. Implement the forward pass for both the action and the state value
#         """
        
#         ########## YOUR CODE HERE (3~5 lines) ##########
#         # calculate input layer
#         state = torch.Tensor(state)
#         output = self.layer1(state)
#         output = self.relu1(output)

#         # calculate policy network
#         action_prob = self.action_layer2(output)
#         action_prob = self.action_relu2(action_prob)
#         action_prob = self.action_layer3(action_prob)
#         action_prob = self.action_softmax(action_prob)

#         # calculate vfa network
#         state_value = self.value_layer2(output)
#         state_value = self.value_relu2(state_value)
#         state_value = self.value_layer3(state_value)
#         ########## END OF YOUR CODE ##########
        
#         return action_prob, state_value

#     def calculate_loss(self, rewards, gamma=0.999):
#         """
#             Calculate the loss (= policy loss + value loss) to perform backprop later
#             TODO:
#                 1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
#                 2. Calculate the policy loss using the policy gradient
#                 3. Calculate the value loss using either MSE loss or smooth L1 loss
#         """
        
#         # Initialize the lists and variables
#         R = 0
#         saved_actions = self.saved_actions
#         returns = []
#         policy_losses = []
#         value_losses = []
#         eps = np.finfo(np.float32).eps.item()

#         ########## YOUR CODE HERE (8-15 lines) ##########
#         for r in rewards[::-1]:
#             R = r + gamma * R
#             returns.insert(0, R)

#         returns = torch.tensor(returns)
#         returns = (returns - returns.mean()) / (returns.std() + eps)
        
#         state_values = torch.stack([values for (log_prob, values) in saved_actions]).squeeze()
#         action_probs = torch.stack([log_prob for (log_prob, values) in saved_actions])

#         ### Uncomment for reinforce with baseline ###
#         # estimator = returns - state_values         

#         ### Uncomment for reinforce vanilla ###
#         estimator = returns

#         ### Uncomment for reinforce with GAE ###     
#         policy_losses = -torch.sum(action_probs * estimator)
#         value_losses = torch.sum((state_values - estimator)**2)
#         ########## END OF YOUR CODE ##########

#         return value_losses + policy_losses

#     def clear_memory(self):
#         # reset rewards and action buffer
#         del self.rewards[:]
#         del self.saved_actions[:]



K = 20
N = 10
action_space = (2 * N) + 1
variance = 10
HINDSIGHT_SIZE = 5

value_function = torch.nn.Sequential(torch.nn.Linear(K + 1, 1))
policy_function = torch.nn.Sequential(torch.nn.Linear(K + 1, action_space))
hindsight_function = GRUNet(input_dim=K, hidden_dim=32, output_dim=HINDSIGHT_SIZE, n_layers=32)
hindsight_classifier = torch.nn.Sequential(
                                torch.nn.Linear(HINDSIGHT_SIZE, 32),
                                torch.nn.ReLU(),
                                torch.nn.Linear(32, 32),
                                torch.nn.ReLU(),
                                torch.nn.Linear(32, action_space))

env = Bandit(K=K, N=N, variance=variance)
num_iterations = 100

def forward_gru():

feedback, context, reward, _ = env.init()
for iterations in range(num_iterations):
    
    feedback_context = np.concatenate(feedback, np.array([context]))

    action = policy_function(feedback_context)
    value = value_function(feedback_context)
    hindsight = hindsight_function(feedback)

    fcb = hindsight_classifier(hindsight, feedback) # future conditional baseline

    loss = calculate_loss(reward, )
    