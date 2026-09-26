import torch
from torch import nn

class ActorCriticNet(nn.Module):
    """Independent linear heads on one-hot states: tabular neural AC.

    Values are exposed in original reward units; fitting uses reward_scale.
    """
    def __init__(self, cfg):
        super().__init__()
        self.actor_head = nn.Linear(cfg.net.input_dim,cfg.net.action_dim,bias=False)
        self.critic_head = nn.Linear(cfg.net.input_dim,1,bias=False)
        nn.init.zeros_(self.actor_head.weight)
        nn.init.zeros_(self.critic_head.weight)
        self.register_buffer("nonterminal",torch.tensor((cfg.env.grid.ravel()!=2).astype('float32')))
        self.reward_scale = cfg.reward_scale

    def forward(self, state):
        logits = self.actor_head(state)
        value = self.critic_head(state).squeeze(-1)*self.reward_scale
        return logits,value*(state@self.nonterminal)
