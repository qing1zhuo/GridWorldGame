import torch
from torch import nn
from config import *
class ActorCriticNet(nn.Module):
    def __init__(self,cfg:config):
        super().__init__()
        self.input_dim=cfg.net.input_dim
        self.action_dim=cfg.net.action_dim
        self.hidden_dim=cfg.net.hidden_dim
        self.backbone_nums=len(self.hidden_dim)

        layers=[]
        layers.append(nn.Linear(self.input_dim,self.hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(self.backbone_nums-1):
            layers.append(nn.Linear(self.hidden_dim[i],self.hidden_dim[i+1]))
            layers.append(nn.ReLU())

        self.backbone_model=nn.Sequential(*layers)
        self.actor_head=nn.Linear(self.hidden_dim[-1],self.action_dim)
        self.critic_head=nn.Linear(self.hidden_dim[-1],1)
    def forward(self,state):
        features=self.backbone_model(state)
        action_score=self.actor_head(features)
        state_value=self.critic_head(features).squeeze(-1)

        return action_score,state_value
