import numpy as np
import matplotlib.pyplot as plt
import torch

class Bandit():
   def __init__(self, K=20, N=100, variance=10, seed=1):
      self.K = K
      self.N = N
      self.ACTION_DIM = int((2 * N) + 1)
      self.prev_context = None
      self.variance = variance

      np.random.seed(seed)
      self.U = np.random.randn(K, self.ACTION_DIM)
      self.V = np.random.randn(K, self.ACTION_DIM)
      self.W = np.random.randn(K, 1)
      self.number_range = np.array([num for num in range(-N, N+1)])

   def selection_index(self, value):
      return np.where(self.number_range == value)[0]
   
   def calculate_feedback(self, context, action, epsilon, feedback_type):
      if feedback_type == "hindsight":
         return self.U[:, self.selection_index(context)] + self.V[:, self.selection_index(action)]
      else:
         return self.U[:, self.selection_index(context)]


   def reset(self, feedback_type):
      C = np.random.choice(self.number_range)
      random_action = np.random.choice(self.number_range)
      epsilon = np.random.normal(loc=0, scale=self.variance)

      feedback_vector = self.calculate_feedback(C, random_action, epsilon, feedback_type).reshape(-1)
      self.prev_context = C

      state = torch.cat((torch.Tensor(feedback_vector), torch.Tensor([C])), 0)
      # state = torch.Tensor([C])
      return state

   def step(self, action, feedback_type):
      reward = -np.power((self.prev_context - action), 2)

      C = np.random.choice(self.number_range)
      epsilon = np.random.normal(loc=0, scale=self.variance)
      feedback_vector = self.calculate_feedback(C, action, epsilon, feedback_type).reshape(-1)
      done = True

      self.prev_context = C
      next_state = torch.cat((torch.Tensor(feedback_vector), torch.Tensor([C])), 0)
      # next_state = torch.Tensor([C])

      return next_state, reward, done

   # def reset(self):
   #    self.U = np.random.randn(self.K, self.N)
   #    self.V = np.random.randn(self.K, self.N)
   #    self.W = np.random.randn(self.K, self.N)

    

