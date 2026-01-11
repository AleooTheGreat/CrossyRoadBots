import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents.PPO.ActorCritic import ActorCritic

class PPO:
    def __init__(self, state_dim, action_dim, lr = 3e-4, gamma = 0.99, eps_clip = 0.2, K = 5, gae_lambda=0.95, hidden_dim=256):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K = K
        self.gae_lambda = gae_lambda
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr)
        
        self.old_policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'done': [],
            'values': []
        }
        
    
    def select_action(self, state):
        
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            action, action_log_prob, state_value = self.old_policy.act(state)
    
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['log_probs'].append(action_log_prob)
        self.buffer['values'].append(state_value)
        
        return action

    def store_transition(self, reward, done):
        
        self.buffer['rewards'].append(reward)
        self.buffer['done'].append(done)
        
    def compute_gae(self, rewards, value, done, next_value):
        
        adv = []
        gae = 0
        
        value = value + [next_value]
        
        for t in reversed(range(len(rewards))):
            
            delta = rewards[t] + self.gamma * value[t + 1] * (1 - done[t])- value[t]
            
            gae = delta + self.gamma * self.gae_lambda * (1 - done[t]) * gae
            
            adv.insert(0, gae)
        
        r = [a + v for a, v in zip(adv, value[:-1])]
        return adv, r
    
    def update(self):
        
        if len(self.buffer['states']) == 0:
            return
        
        with torch.no_grad():
            last_state = self.buffer['states'][-1]
            _, next_value = self.old_policy.forward(last_state)
            next_value = next_value.item() if not self.buffer['done'][-1] else 0
            
            value_list = [v.item() for v in self.buffer['values']]
            adv, r = self.compute_gae(
                self.buffer['rewards'],
                value_list,
                self.buffer['done'],
                next_value
            )
            
        old_state = torch.stack(self.buffer['states']).detach().to(self.device)
        old_action = torch.tensor(self.buffer['actions']).detach().to(self.device)
        old_log_prob = torch.stack(self.buffer['log_probs']).detach().to(self.device)
        adv = torch.tensor(adv, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for e in range(self.K):
            log_prob, state_value, h = self.policy.evaluate(old_state, old_action)
            
            ratio = torch.exp(log_prob - old_log_prob.detach())
            
            ppo = -torch.min(
                ratio * adv, 
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                )
            
            actor_loss = ppo.mean()
            critic_loss = 0.5 * self.MseLoss(state_value.squeeze(), r)
            h_loss = -0.01 * h.mean()
            
            Loss = actor_loss + critic_loss + h_loss
            
            self.optimizer.zero_grad()
            Loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'done': [],
            'values': []
        }
        
    def save(self, filepath):
        
        torch.save(
        {
        'policy_state_dict': self.policy.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
        }, 
        filepath)
    
    def load(self, filepath):
        
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.old_policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])