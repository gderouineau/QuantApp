# neuronal_strategy/services/torch_models.py
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int | None = None, dropout: float = 0.1,
                 layers: list[int] | None = None):
        super().__init__()
        if layers is None:
            if hidden is None:
                hidden = 128
            layers = [hidden]

        dims = [input_dim] + layers + [1]
        mods = []
        for i in range(len(dims) - 2):  # toutes les couches cachées
            mods.append(nn.Linear(dims[i], dims[i+1]))
            mods.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                mods.append(nn.Dropout(dropout))
        mods.append(nn.Linear(dims[-2], dims[-1]))  # couche de sortie (logit)

        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (N,) logits
