import torch
import torch.nn as nn
import torchvision.transforms as transforms

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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
          )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hindsight, policy_action_dist):
        action_dist = self.net(hindsight) + torch.log(policy_action_dist)
        return self.softmax(action_dist)

class LSTMNet(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, device, n_layers, drop_prob=0.0):
    super(LSTMNet, self).__init__()
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.device = device
        
    self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=False, dropout=drop_prob)
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.relu = nn.ReLU()
        
  def forward(self, x, h):
    out, h = self.lstm(x, h)
    out = self.fc(self.relu(out))
    return out, h
    
  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    if batch_size == 1:
        hidden = (weight.new(self.n_layers, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, self.hidden_dim).zero_().to(self.device))
    else:
      hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
              weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))
    return hidden

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.0):
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

class ConvNet(nn.Module):
    def __init__(self, output_size):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        - A 2D convolutional layer (torch.nn.Conv2d) with 7 output channels, with kernel size 3
        - A 2D maximimum pooling layer (torch.nn.MaxPool2d), with kernel size 2
        - A 2D convolutional layer (torch.nn.Conv2d) with 3 output channels, with kernel size 2
        - A fully connected (torch.nn.Linear) layer with 10 output features

        '''
        super(ConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!

        # Please ONLY define the sub-modules here
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.batch1 = torch.nn.BatchNorm2d(16)
        self.rel1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.batch2 = torch.nn.BatchNorm2d(32)
        self.rel2 = torch.nn.ReLU()
        
        self.lin = torch.nn.Linear(9687552, output_size)

    def forward(self, image):
        transform = transforms.Compose([transforms.PILToTensor()])
        x = transform(image).float()
        x = self.conv1(x)
        # x = self.batch1(x)
        x = self.rel1(x)

        x = self.conv2(x)
        # x = self.batch2(x)
        x = self.rel2(x)

        x = x.reshape((1, x.shape[0] * x.shape[1] * x.shape[2]))
        x = self.lin(x)
        return x