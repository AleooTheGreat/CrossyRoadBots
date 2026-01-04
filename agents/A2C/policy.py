import torch


class PolicyNet(torch.nn.Module):
    def __init__(self, input_dim: int = 8, action_dim: int = 5):
        super(PolicyNet, self).__init__()

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), # Mărit de la 128
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),       # Mărit de la 64
            torch.nn.ReLU(),
            torch.nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.policy(x)