import numpy as np
#import matplotlib.pyplot as plt
import torch

class Bandit():
   def __init__(self, K=20, N=100, variance=10, uvw_scale=100, seed=None):
      self.K = K
      self.N = N
      self.ACTION_DIM = int((2 * N) + 1)
      self.prev_context = None
      self.variance = variance

      self.rng = np.random.default_rng(seed)
      self.U = self.rng.normal(scale=uvw_scale, size=(K, self.ACTION_DIM))
      self.V = self.rng.normal(scale=uvw_scale, size=(K, self.ACTION_DIM))
      self.W = self.rng.normal(scale=uvw_scale, size=K)

   def sample(self):
      return self.rng.integers(-self.N, self.N + 1).item()

   def calculate_feedback(self, C, A, epsilon):
      return self.U[:, C + self.N] + self.V[:, A + self.N] + self.W * epsilon

   def step(self, action_index):
      action = action_index - self.N  # cast [0, ..., 20] to [-10, ..., 10]
      epsilon = self.rng.normal(loc=0, scale=self.variance)
      reward = -np.power((self.prev_context - action), 2) + epsilon
      feedback_vector = self.calculate_feedback(self.prev_context, action, epsilon)
      done = True

      C = self.sample()
      self.prev_context = C
      next_state = (feedback_vector, C)

      return next_state, reward, done

   #
   def reset(self):
      C = self.sample()
      self.prev_context = C
      return C


    

