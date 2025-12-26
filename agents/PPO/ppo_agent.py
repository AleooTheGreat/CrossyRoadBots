import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
    
    def act(self, state):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, state_value
    
    def evaluate(self, states, actions):
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4, gae_lambda=0.95, hidden_dim=256):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.buffer = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }
        
    def select_action(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_value = self.policy_old.act(state)
        
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['logprobs'].append(action_logprob)
        self.buffer['values'].append(state_value)
        
        return action
    
    def store_transition(self, reward, done):
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns
    
    def update(self):
        if len(self.buffer['states']) == 0:
            return
        
        with torch.no_grad():
            last_state = self.buffer['states'][-1]
            _, next_value = self.policy_old.forward(last_state)
            next_value = next_value.item() if not self.buffer['dones'][-1] else 0.0
        
        values_list = [v.item() for v in self.buffer['values']]
        advantages, returns = self.compute_gae(
            self.buffer['rewards'], 
            values_list, 
            self.buffer['dones'], 
            next_value
        )
        
        old_states = torch.stack(self.buffer['states']).detach().to(self.device)
        old_actions = torch.tensor(self.buffer['actions']).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer['logprobs']).detach().to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.MseLoss(state_values.squeeze(), returns)
            entropy_loss = -0.01 * dist_entropy.mean()
            
            loss = actor_loss + critic_loss + entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }
    
    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
