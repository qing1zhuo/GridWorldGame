import torch
from torch import nn

class main_net(nn.Module):
    def __init__(
        self,
        input_dim=2,output_dim=5,
        hidden_dim=(20,10)
    ):
        super().__init__()
        if not hidden_dim:
            raise ValueError("hidden_dim must contain at least one layer size")
        num=hidden_dim.__len__()+1
        layers=[]

        layers.append(nn.Linear(input_dim,hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(num-2):
            layers.append(nn.Linear(hidden_dim[i],hidden_dim[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1],output_dim))

        self.model=nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

class target_net(main_net):
    """Target network with exactly the same architecture as the main network."""

    pass
