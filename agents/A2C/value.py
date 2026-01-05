import torch

class ValueNet(torch.nn.Module):
    def __init__(self, input_dim: int = 8):
        super(ValueNet, self).__init__()

        self.value = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.value(x)