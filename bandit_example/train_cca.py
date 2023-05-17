from bandit import Bandit
from models import GRUNet, VFA, Policy, HindsightClassifier

import torch
from torch.distributions import Categorical
from collections import namedtuple
import torch.optim as optim

SavedAction = namedtuple('SavedAction', ['policy_log_prob', 'value', 'hindsight_log_prob'])

class ModelHolder():
    def __init__(self, env, hidden_size=32, lr_hs=0.001, lr_im=0.001, lr_sup=0.001):
        FEEDBACK_DIM = env.K
        HINDSIGHT_DIM = FEEDBACK_DIM
        HIDDEN_SIZE = hidden_size
        

        self.saved_actions = []
        self.rewards = []

        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")

        self.hindsight_network = GRUNet(input_dim=FEEDBACK_DIM, hidden_dim=32, 
                        output_dim=HINDSIGHT_DIM, n_layers=32, device=self.device).to(self.device)
        self.hindsight_optimizer = optim.Adam(self.hindsight_network.parameters(), lr=lr_im)
        
        self.vfa = VFA(observation_dim=FEEDBACK_DIM + HINDSIGHT_DIM, hidden_size=HIDDEN_SIZE).to(self.device)
        self.vfa_optimizer = optim.Adam(self.vfa.parameters(), lr=lr_hs)

        self.policy = Policy(observation_dim=1, hidden_size=HIDDEN_SIZE, action_dim=env.ACTION_DIM).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_hs)

        self.hindsight_classifier = HindsightClassifier(observation_dim=FEEDBACK_DIM + HINDSIGHT_DIM, 
                                                hidden_size=HIDDEN_SIZE, output_dim=env.ACTION_DIM).to(self.device)
        self.hs_classifier_optimizer = optim.Adam(self.hindsight_classifier.parameters(), lr=lr_sup)
    
    def select_action(self, state, env):
        # Use Policy P(a | s) to select action
        X = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action_prob_dist = self.policy(X)
        action_prob_dist = Categorical(action_prob_dist)
        action = action_prob_dist.sample()

        # Calculate feedback vector and hindsight
        feedback_vector = torch.from_numpy(env.calculate_feedback(action)).float()
        # h = self.hindsight_network.init_hidden(feedback_vector.shape[0])
        h = self.hindsight_network.init_hidden(batch_size=1)
        hindsight, h = self.hindsight_network(feedback_vector.reshape((1, 1, feedback_vector.size(0))), h)
        hindsight = hindsight.reshape(hindsight.size(1))

        # Calculate state value using hindsight and feedback
        hindsight_observation = torch.cat((hindsight, feedback_vector))
        state_value = self.vfa(hindsight_observation)

        # Find probability of selected action under hindsight classifier P(a | s, psi)
        hindsight_action_prob_dist = self.hindsight_classifier(hindsight_observation)
        hindsight_action_prob_dist = Categorical(hindsight_action_prob_dist)
        hindsight_log_prob = hindsight_action_prob_dist.log_prob(action)

        self.saved_actions.append(SavedAction(action_prob_dist.log_prob(action), state_value, hindsight_log_prob))

        return action.item()

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
            policy_log_prob, state_value, hindsight_log_prob = saved_actions[t]
            advantage = Gt - state_value.detach().item()

            hindsight_baseline_loss.append((state_value[0] - Gt)**2)  # simple MSE loss
            policy_gradient_loss.append(-policy_log_prob * advantage)

            # NOTE: might need to sum over actions
            action_independence_loss.append(policy_log_prob * (policy_log_prob - hindsight_log_prob)) 
            hindsight_classifier_loss.append(-hindsight_log_prob)

        hindsight_baseline_loss = torch.stack(hindsight_baseline_loss).sum()
        hindsight_classifier_loss = torch.stack(hindsight_classifier_loss).sum()
        action_independence_loss = torch.stack(action_independence_loss).sum()
        policy_gradient_loss = torch.stack(policy_gradient_loss).sum()

        return hindsight_baseline_loss, hindsight_classifier_loss, action_independence_loss, policy_gradient_loss


    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]
    
    def update(self):
        L_hs, L_sup, L_im, L_pg = self.calculate_loss()

        # Optimize VFA (Hindsight Baseline Loss)
        self.vfa_optimizer.zero_grad()
        L_hs.backward()
        self.vfa_optimizer.step()

        # Optimize Hindsight Network (Independence Maximization Loss)
        self.policy_optimizer.zero_grad()
        self.hindsight_optimizer.zero_grad()
        L_im.backward()
        self.hindsight_optimizer.step()
        self.policy_optimizer.step()

        # Optimize Hindsight Classifier (Hindsight Predictor Loss)
        self.hs_classifier_optimizer.zero_grad()
        L_sup.backward()
        self.hs_classifier_optimizer.step()

        # Optimize Policy (Policy Gradient Loss)
        self.policy_optimizer.zero_grad()
        L_pg.backward()
        self.policy_optimizer.step()

        self.clear_memory()

def train(env):
    MAX_EPISODES = 100000
    MAX_EPISODE_LENGTH = 1000
    EWMA_MOVE = 0.0005
    EWMA_THRESHOLD = -1

    model = ModelHolder(env)
    
    for i_episode in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0
        done = False

        for t in range(MAX_EPISODE_LENGTH):
            action = model.select_action(state, env)
            state, reward, done = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward

            if done:
                break
        
        model.update()

        # update EWMA reward and log the results
        ewma_reward = EWMA_MOVE * ep_reward + (1 - EWMA_MOVE) * ewma_reward
        if i_episode % 10 == 0:
            print('Episode {}\treward: {}\t ewma reward: {}'.format(i_episode, ep_reward, ewma_reward))

        if i_episode > 10000 and ewma_reward > EWMA_THRESHOLD:
            print("Solved in Episode {}! Running reward is now {}!".format(i_episode, ewma_reward))
            break
    

def main():
    random_seed = 1
    env = Bandit(K=20, N=10, variance=0.0, seed=random_seed)
    torch.manual_seed(random_seed)
    train(env)

if __name__ == '__main__':
    main()