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

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size) #output dimension = action size != action space size, da action space size nicht mehr in dem sinn existiert (action ist immer ein Vektor, der alle dimensionen enthaelt)
        #Beispiel: 3 Motoren -> Action = (0.3, 1.5, 5.6) es werden immer alle Freiheitsgrade bedient und deren Magnitude-Werte ausgegeben (z.b. Motor 1 auf 30%)
        #the output is wohlgemerkt a vector containing a value for each dimension of the proposed action.
        #Always, all action dimensions are predicted, because all actions are used each prediction. In order not to do a single action, its magnitude (vector-element-value) will just be 0
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)#input size = state_size, neurons = fcs1_units
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units) #number of nodes is higher, the higher the number of dimensions of each action is
        self.fc3 = nn.Linear(fc2_units, 1) #the output is the state-action value (expected reward when choosing this action vector in current state) for the action_dimension_values vector returned by the actor
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)# neuron output of first layer is concatenated with the action vector (each element is one dimension of the action) the value of each action dimension is the continuous value for the
        #continuous action space
        x = F.relu(self.fc2(x))
        return self.fc3(x)
