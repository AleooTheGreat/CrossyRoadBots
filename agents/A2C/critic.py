import torch

class Critic(torch.nn.Module):
    def __init__(self, input_dim: int = 49):
        super(Critic, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x: torch.Tensor):
        return self.layers(x)