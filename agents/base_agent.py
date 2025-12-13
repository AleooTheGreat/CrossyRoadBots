from abc import ABC, abstractmethod
import os

class BaseAgent(ABC):
    
    def __init__(self, agent_name, state_size, action_size, level='medium'):
        self.agent_name = agent_name
        self.state_size = state_size
        self.action_size = action_size
        self.level = level
        
        self.results_dir = f'results/{agent_name}'
        self.checkpoint_dir = f'checkpoints/{agent_name}'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    @abstractmethod
    def select_action(self, state, training=True):
        pass
    
    @abstractmethod
    def train_step(self, state, action, reward, next_state, done):
        pass
    
    @abstractmethod
    def save(self, filepath):
        pass
    
    @abstractmethod
    def load(self, filepath):
        pass
    
    def get_checkpoint_path(self, episode):
        return os.path.join(self.checkpoint_dir, f'checkpoint_ep_{episode}.pth')
