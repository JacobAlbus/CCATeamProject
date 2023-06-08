from grid_world import GridWorld
from models import GRUNet, VFA, Policy, HindsightClassifier, ConvNet, LSTMNet

import torch
from torch.distributions import Categorical
from collections import namedtuple
import torch.optim as optim
import cv2
import PIL
from torch.utils.tensorboard import SummaryWriter

# Define a tensorboard writer
writer = SummaryWriter("./logs")
SavedAction = namedtuple('SavedAction', ['value', 'action', 'hindsight_observation', 
                                         'policy_log_dist', 'hindsight_log_dist'])

class ModelHolder():
    def __init__(self, env, hidden_size=32, lr_hs=0.0005, lr_im=0.0005, lr_sup=0.0005):
        CONV_OUTPUT_SIZE = 128
        AGENT_STATE_SIZE = 16
        HINDSIGHT_SIZE = 128
        HIDDEN_SIZE = hidden_size
        torch.autograd.set_detect_anomaly(True)

        self.saved_actions = []
        self.rewards = []

        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")

        self.hindsight_network = GRUNet(input_dim=AGENT_STATE_SIZE + 1, hidden_dim=32, 
                                        output_dim=HINDSIGHT_SIZE, n_layers=1, device=self.device).to(self.device)
        self.hindsight_optimizer = optim.Adam(self.hindsight_network.parameters(), lr=lr_im)
        self.hindsight_hidden_state = self.hindsight_network.init_hidden(batch_size=1)

        self.ConvNet = ConvNet(output_size=CONV_OUTPUT_SIZE)

        self.agent_state_lstm = LSTMNet(input_dim=CONV_OUTPUT_SIZE + 1, hidden_dim=128, 
                                        output_dim=AGENT_STATE_SIZE, device=self.device, n_layers=1).to(self.device)
        self.agent_state_lstm_hidden_state = self.agent_state_lstm.init_hidden(batch_size=1)
        
        self.vfa = VFA(observation_dim=AGENT_STATE_SIZE, hidden_size=HIDDEN_SIZE).to(self.device)
        self.vfa_optimizer = optim.Adam(self.vfa.parameters(), lr=lr_hs)

        self.hindsight_baseline = VFA(observation_dim=AGENT_STATE_SIZE + HINDSIGHT_SIZE, hidden_size=HIDDEN_SIZE).to(self.device)

        self.policy = Policy(observation_dim=AGENT_STATE_SIZE, hidden_size=64, action_dim=env.ACTION_DIM).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_hs)

        self.hindsight_classifier = HindsightClassifier(observation_dim = AGENT_STATE_SIZE + HINDSIGHT_SIZE, 
                                                        hidden_size=256, output_dim=env.ACTION_DIM).to(self.device)
        self.hs_classifier_optimizer = optim.Adam(self.hindsight_classifier.parameters(), lr=lr_sup)
    
    def select_action(self, env_img, reward, env):
        # Use CNN to vectorize image of environment
        cnn_output = self.ConvNet(env_img)
        reward = torch.Tensor([reward]).reshape((1, 1))
        cnn_output_reward = torch.cat((cnn_output, reward), dim=-1)

        # Calculate agent state
        agent_state, hidden_state = self.agent_state_lstm(cnn_output_reward, self.agent_state_lstm_hidden_state)
        self.agent_state_lstm_hidden_state = hidden_state

        # Calculate policy distribution P(a | s)
        policy_action_dist = self.policy(agent_state)

        # Calculate and hindsight
        agent_state_reward = torch.cat((agent_state, reward), dim=-1)
        agent_state_reward = agent_state_reward.reshape((1, 1, agent_state_reward.size(1)))
        hindsight, hidden_state = self.hindsight_network(agent_state_reward,self.hindsight_hidden_state)
        hindsight = hindsight.reshape(hindsight.size(1))
        self.hindsight_hidden_state = hidden_state

        # Calculate hindsight baseline
        hindsight_agent_state = torch.cat((hindsight, agent_state.flatten()), dim=-1)
        hindsight_baseline = self.hindsight_baseline(hindsight_agent_state) + self.vfa(agent_state)

        # Get probability distribution of action conditioned on feedback and hindsight
        # detach hindsight to prevent gradient being applied to hindsight RNN/GRU
        hindsight_action_dist = self.hindsight_classifier(hindsight_agent_state.detach(), policy_action_dist)
        hindsight_action_dist = Categorical(hindsight_action_dist)

        # Select action
        policy_action_categorical = Categorical(policy_action_dist)
        action = policy_action_categorical.sample()

        # save state value, selected action, prob. dist. of action conditioned on obersvation, and 
        # prob. dist. of action conditioned on feedback and hindsight 
        self.saved_actions.append(SavedAction(hindsight_baseline, action, hindsight_agent_state, policy_action_dist, hindsight_action_dist))

        return action.item()
    
    def calculate_action_independence_loss(self, policy_action_dist, hindsight_agent_state):
        hindsight_action_dist = self.hindsight_classifier(hindsight_agent_state, policy_action_dist)
        hindsight_action_dist = Categorical(hindsight_action_dist)
        loss = []

        policy_action_categorical = Categorical(policy_action_dist)
        for action in policy_action_categorical.enumerate_support():

            # detach policy network values to prevent gradient being applied to policy network
            policy_prob = policy_action_categorical.probs[:, action.item()].detach() 
            policy_log_prob = policy_action_categorical.log_prob(action).detach()
            hindsight_log_prob = hindsight_action_dist.log_prob(action)
            loss.append(policy_prob * (policy_log_prob - hindsight_log_prob))
        
        loss = torch.stack(loss)
        return loss.sum()
    
    def calculate_hindsight_classifier_loss(self, hindsight_prob_dist, hindsight_log_prob):
        # loss = torch.Tensor(0)
        loss = []

        for action in hindsight_prob_dist.enumerate_support():
            loss.append(hindsight_log_prob * hindsight_prob_dist.probs[:, action.item()])
        
        loss = torch.stack(loss)
        return -loss.sum()

    def calculate_loss(self, gamma=0.99):
        # Initialize the lists and variables
        Gt = 0.0
        saved_actions = self.saved_actions
        hindsight_baseline_loss = []
        hindsight_classifier_loss = []
        action_independence_loss = []
        policy_gradient_loss = []

        for t in reversed(range(len(self.rewards))):
            Gt = Gt * gamma + self.rewards[t]
            hindsight_baseline, action, hindsight_agent_state, policy_action_dist,  hindsight_action_dist = saved_actions[t]

            policy_log_prob = Categorical(policy_action_dist).log_prob(action)

            # Caluclate MSE baseline loss
            hindsight_baseline_loss.append((hindsight_baseline - Gt)**2)

            # Detach state value to prevent gradient being applied to VFA
            policy_gradient_loss.append(-policy_log_prob * (Gt - hindsight_baseline.detach().item()))

            action_independence_loss.append(self.calculate_action_independence_loss(policy_action_dist, hindsight_agent_state))

            hindsight_log_prob = hindsight_action_dist.log_prob(action)
            hindsight_classifier_loss.append(self.calculate_hindsight_classifier_loss(hindsight_action_dist, hindsight_log_prob))

        hindsight_baseline_loss = torch.stack(hindsight_baseline_loss).sum()
        hindsight_classifier_loss = torch.stack(hindsight_classifier_loss).sum()
        action_independence_loss = torch.stack(action_independence_loss).sum()
        policy_gradient_loss = torch.stack(policy_gradient_loss).sum()

        return hindsight_baseline_loss, hindsight_classifier_loss, action_independence_loss, policy_gradient_loss


    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]
        self.hindsight_hidden_state = self.hindsight_network.init_hidden(batch_size=1)
        self.agent_state_lstm_hidden_state = self.forward_lstm.init_hidden(batch_size=1)

    
    def update(self, max_timestep):
        L_hs, L_sup, L_im, L_pg = self.calculate_loss()
        print("calculated")
        self.vfa_optimizer.zero_grad()
        self.hindsight_optimizer.zero_grad()
        self.hs_classifier_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        print("zero grad")

        L_hs.backward(retain_graph=False)         # Optimize VFA (Hindsight Baseline Loss)
        L_im.backward(retain_graph=False)         # Optimize Hindsight Network (Independence Maximization Loss)
        L_sup.backward(retain_graph=False)        # Optimize Hindsight Classifier (Hindsight Predictor Loss)
        L_pg.backward()         # Optimize Policy (Policy Gradient Loss)

        self.vfa_optimizer.step()
        self.hindsight_optimizer.step()
        self.hs_classifier_optimizer.step()
        self.policy_optimizer.step()

        self.clear_memory()

def train(env):
    MAX_EPISODES = 10000000
    MAX_EPISODE_LENGTH = 50
    EWMA_MOVE = 0.005
    EWMA_THRESHOLD = -0.1

    model = ModelHolder(env)
    ewma_reward = 0
    
    for i_episode in range(MAX_EPISODES):
        state = env.reset()
        reward = 0
        ep_reward = 0
        done = False

        for t in range(MAX_EPISODE_LENGTH):
            env.show(0)
            img = cv2.imread("temp/env_image.png")
            env_image = PIL.Image.fromarray(img)

            action = model.select_action(env_image, reward, env)
            state, reward, done = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward

            if done:
                break
        
        model.update(t)
        print("updated")

        # update EWMA reward and log the results
        ewma_reward = EWMA_MOVE * ep_reward + (1 - EWMA_MOVE) * ewma_reward
        if i_episode % 1 == 0:
            print('Episode {}\treward: {}\t ewma reward: {}'.format(i_episode, ep_reward, ewma_reward))

        if i_episode > 10000 and ewma_reward > EWMA_THRESHOLD:
            print("Solved in Episode {}! Running reward is now {}!".format(i_episode, ewma_reward))
            break
        
        writer.add_scalar("EWMA Reward", ewma_reward, i_episode)
        writer.add_scalar("Episode Reward", ep_reward, i_episode)

def main():
    random_seed = 1
    env = GridWorld(low_variance=False, seed=random_seed)
    torch.manual_seed(random_seed)
    train(env)

if __name__ == '__main__':
    main()