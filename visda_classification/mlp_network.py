import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, n_feature, out_dim):
        super(MLP, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_feature),
            torch.nn.ReLU(),
            torch.nn.Linear(n_feature, n_feature),
            torch.nn.ReLU(),
            torch.nn.Linear(n_feature, out_dim)
        )

    def forward(self, x):
        x = self.layer(x)
        return x
