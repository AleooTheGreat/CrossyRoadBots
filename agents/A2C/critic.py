import torch

class Critic(torch.nn.Module):
    def __init__(self, input_dim: int = 49):
        super(Critic, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)

    def forward(self, x: torch.Tensor):
        return self.layers(x)