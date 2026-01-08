import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        
        action_prob = self.actor(state)
        state_value = self.critic(state)
        
        return action_prob, state_value
    
    def act(self, state):
        
        action_prob, state_value = self.forward(state)
        
        distribution = Categorical(action_prob)
        
        action = distribution.sample()
        action_log = distribution.log_prob(action)
        
        return action.item(), action_log, state_value
        
    def evaluate(self, states, actions):
        
        action_prob, state_value = self.forward(states)
        
        distribution = Categorical(action_prob)
        
        action_log = distribution.log_prob(actions)
        h = distribution.entropy()
        
        return action_log, state_value, h