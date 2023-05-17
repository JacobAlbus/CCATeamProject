import torch
import torch.nn as nn

# is_cuda = torch.cuda.is_available()

# # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
# if is_cuda:
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

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

class HindsightClassifier(nn.Module):
    def __init__(self, observation_dim, hidden_size, output_dim):
        super(HindsightClassifier, self).__init__()
        self.double()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Softmax(dim=-1)
    )
    def forward(self, input):
        action_dist = self.net(input)
        return action_dist

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
