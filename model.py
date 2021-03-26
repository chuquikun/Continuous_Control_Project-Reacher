import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[64,128]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers ([int]): A list with the dimension of the inner layers, the first element
            is the output size after pass the state
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        hl = hidden_layers.copy()
        hl.insert(0,state_size)
        hl.append(action_size)
        self.layer_sizes = zip(hl[:-1], hl[1:])
        self.layers = nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in self.layer_sizes])
        self.reset_parameters()
        
        
    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.leaky_relu(self.layers[0](state))
        for layer in self.layers[1:-1]:
            x = F.leaky_relu(layer(x))
        return F.tanh(self.layers[-1](x))

    
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[64,128]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers ([int]): A list with the dimension of the inner layers, the first element
            is the output size after pass the state
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        hl = hidden_layers.copy()
        hl.insert(1,hl[0]+action_size)
        hl.insert(0,state_size)
        hl.append(1)
        layer_sizes = zip(hl[2:-1], hl[3:])
        #The network is comprised by a series of fully conected layers with a single scalar value output
        self.layers = nn.ModuleList([nn.Linear(hl[0], hl[1])])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.reset_parameters()

        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.layers[0](state))
        x = torch.cat((xs, action), dim=1)
        for layer in self.layers[1:-1]:
             x = F.leaky_relu(layer(x))
        return self.layers[-1](x)
