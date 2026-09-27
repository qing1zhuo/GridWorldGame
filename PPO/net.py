import torch 
from torch import nn

class OneHotEncoding(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes=num_classes

    def forward(self,x):
        return torch.nn.functional.one_hot(
            x.long(),num_classes=self.num_classes
        ).float()

class Actor(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.input_dim=cfg.input_dim
        self.action_dim=cfg.action_dim
        self.actor_hidden_dim=cfg.actor_hidden_dim

        nums=len(self.actor_hidden_dim)
        layers=[OneHotEncoding(self.input_dim)]
        layers.append(nn.Linear(self.input_dim,self.actor_hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(nums-1):
            layers.append(nn.Linear(self.actor_hidden_dim[i],self.actor_hidden_dim[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.actor_hidden_dim[-1],self.action_dim))

        self.model=nn.Sequential(*layers)
    def forward(self,x):
        """
        输入: state tensor [batch]
        输出: action_logits tensor [batch,actor_dim]
        """
        return self.model(x)

class Critic(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.input_dim=cfg.input_dim
        self.critic_hidden_dim=cfg.critic_hidden_dim

        nums=len(self.critic_hidden_dim)
        layers=[OneHotEncoding(self.input_dim)]
        layers.append(nn.Linear(self.input_dim,self.critic_hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(nums-1):
            layers.append(nn.Linear(self.critic_hidden_dim[i],self.critic_hidden_dim[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.critic_hidden_dim[-1],1))

        self.model=nn.Sequential(*layers)
    def forward(self,x):
        """
        输入: state tensor [batch]
        输出: state_value tensor [batch,1]
        """
        return self.model(x)
