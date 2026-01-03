import torch


class Actor(torch.nn.Module):
    def __init__(self, input_dim: int = 49, action_dim: int = 5):
        super(Actor, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Softmax(dim=-1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    
    def forward(self, x: torch.Tensor):
        policy = self.layers(x)
        return policy

    def select_action(self, state: torch.Tensor):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)
