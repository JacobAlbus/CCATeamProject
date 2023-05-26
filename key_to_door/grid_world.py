import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from enum import IntEnum

class Pixel(IntEnum):
   EMPTY, WALL, AGENT, KEY, DOOR, APPLE = range(6)
colors = ['white', 'black', 'red', 'purple', 'blue', 'green']  # for matplotlib

# Init with low_variance True to fix the apple reward to 1. Init with False to use random apple reward of 1 or 10.
# Fixed to use a partial field of view of size 5
class GridWorld():
   def __init__(self, low_variance=True, seed=None):
      # constant between episodes
      self.rng = np.random.default_rng(seed)
      self.ACTION_DIM = 4
      self.OBSERVATION_DIM = 26  # room number + 25 grid values
      self.pixels = [(5, 5), (5, 9), (5, 5)]
      self.door_reward = 1
      self.low_variance = low_variance

      # constant through episode
      self.apple_reward = 1
      self.future_agent_pos = [(0, 0), (0, 0)]  # will be sampled

      # variables through episode
      self.agent_room = 0
      self.agent_pos = (0,0)  # will be sampled
      self.agent_has_key = False
      self.num_picked_apples = 0  # part of obs
      self.rooms = []


   def reset(self):
      self.rooms = self.sample_rooms()
      self.agent_room = 0
      self.agent_has_key = False
      self.num_picked_apples = 0
      if self.low_variance:
         self.apple_reward = 1
      else:
         self.apple_reward = self.rng.choice([1, 10])

      i, j = self.agent_pos
      obs = self.rooms[0][i-2:i+3, j-2:j+3]
      return np.insert(obs.flatten(), 0, self.agent_room)


   # visualize the current configuration via matplotlib
   def show(self):
      fig, axs = plt.subplots(1, 3)
      for i, ax in enumerate(axs):
         room = self.rooms[i][1:-1, 1:-1]
         h, l = room.shape
         print(room)
         for i in range(room.shape[0]):
            for j in range(room.shape[1]):
               if room[(i, j)] == Pixel.EMPTY:
                  continue
               rect = mpatches.Rectangle((j / l, (h - i - 1) / h), 1 / l, 1 / h, linewidth=1,
                                         facecolor=colors[room[(i, j)]])
               ax.add_patch(rect)
               ax.set_aspect('equal', adjustable='box')
               ax.set_yticks([])
               ax.set_xticks([])

      # build legend
      handles = []
      for px in Pixel:
         patch = mlines.Line2D([], [], marker="s", markersize=10, linewidth=0, color=colors[px.value], label=px.name)
         handles.append(patch)
      axs[1].legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.5))
      plt.show()

   def sample_rooms(self):
      rooms = []
      for pxs in self.pixels:
         room = np.zeros((pxs[0] + 4, pxs[1] + 4), dtype=np.int32)  # double wall layer
         room[:2, :] = Pixel.WALL
         room[-2:, :] = Pixel.WALL
         room[:, :2] = Pixel.WALL
         room[:, -2:] = Pixel.WALL
         if pxs[0] == pxs[1]:
            room[4, 2] = Pixel.WALL
            room[4, -3] = Pixel.WALL
         rooms.append(room)

      # randomly set agent, key, door
      # room 1
      for itm in [Pixel.AGENT, Pixel.KEY, Pixel.DOOR]:
         idx = self.sample_empty_index(rooms[0])
         if itm == Pixel.AGENT:
            self.agent_pos = idx
         rooms[0][idx] = itm

      # room 2
      for itm in [Pixel.AGENT, Pixel.DOOR] + [Pixel.APPLE for _ in range(10)]:
         idx = self.sample_empty_index(rooms[1])
         if itm == Pixel.AGENT:
            self.future_agent_pos[0] = idx
         rooms[1][idx] = itm

      # room 3
      for itm in [Pixel.AGENT, Pixel.DOOR]:
         idx = self.sample_empty_index(rooms[2])
         if itm == Pixel.AGENT:
            self.future_agent_pos[1] = idx
         rooms[2][idx] = itm

      return rooms

   def sample_empty_index(self, room):
      height, length = room.shape
      i = self.rng.integers(2, height - 2)
      j = self.rng.integers(2, length - 2)
      while room[(i,j)] != Pixel.EMPTY:
         i = self.rng.integers(2, height - 2)
         j = self.rng.integers(2, length - 2)
      return (i,j)

   # for a given action, move the agent, and return the new observation, reward, and done flag
   # actions: 0: top, 1: down, 2: left, 3: right
   # as observation return the current room number and 9 grid values
   def step(self, action_index):
      new_pos = self.agent_pos
      if action_index == 0:
         new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
      elif action_index == 1:
         new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
      elif action_index == 2:
         new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
      elif action_index == 3:
         new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)

      reward = 0
      done = False

      fld = self.rooms[self.agent_room][new_pos]
      # if fld is WALL no movement is performed
      if fld == Pixel.DOOR:  # change room or terminate episode
         if self.agent_room == 2:
            if self.agent_has_key:
               reward = self.door_reward
            done = True
         else:
            self.agent_pos = self.future_agent_pos[self.agent_room]
            self.agent_room += 1

      elif fld in {Pixel.EMPTY, Pixel.KEY, Pixel.APPLE}:  # perform movement
         if fld == Pixel.KEY:
            self.agent_has_key = True
         if fld == Pixel.APPLE:
            reward = self.apple_reward
            self.num_picked_apples += 1
         self.rooms[self.agent_room][self.agent_pos] = Pixel.EMPTY
         self.rooms[self.agent_room][new_pos] = Pixel.AGENT
         self.agent_pos = new_pos

      i, j = self.agent_pos
      obs = self.rooms[self.agent_room][i-2:i+3, j-2:j+3]
      obs = np.insert(obs.flatten(), 0, self.agent_room)

      return obs, reward, done


if __name__ == '__main__':
    gw = GridWorld(seed=10)
    state = gw.reset()
    print(len(state))
    _ = gw.step(2)
    _ = gw.step(2)
    _ = gw.step(1)
    obs, reward, done = gw.step(1)
    print(reward)
    print(obs)
    gw.show()



    

