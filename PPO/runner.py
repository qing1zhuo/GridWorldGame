from env import GridWorld
from net import Actor,Critic
from config import GridWorld_Config
from rollout import GridWorld_Rollout
import torch
from torch import optim
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

class GridWorld_Runner:
    def __init__(self,cfg):
        self.gamma=cfg.gamma
        self.clip_eps=cfg.clip_eps
        self.entropy_coef=cfg.entropy_coef
        self.batch_size=cfg.batch_size
        self.actor=Actor(cfg)
        self.critic=Critic(cfg)

        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=cfg.actor_lr)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=cfg.critic_lr)

        self.rollout=GridWorld_Rollout(cfg)
        self.env=GridWorld(cfg)

    def select_action(self,states:np.ndarray,deterministic=False):
        """
        参数:
            states: ndarray [batch]
            deterministic: bool 是否确定性贪心
        return:
            actions: ndarray [batch]

        """
        states_tensor = torch.as_tensor(states, dtype=torch.long)
        with torch.no_grad():
            action_logits = self.actor(states_tensor)

        if deterministic:
            actions = action_logits.argmax(dim=-1)
            return actions.cpu().numpy().astype(np.int64)
        else:
            dist=Categorical(logits=action_logits)
            actions=dist.sample()
            actions=actions.cpu().numpy().astype(np.int64)
            return actions

    def train_step(self):
        # 这里假定已经rollout过
        batch=self.rollout.sample(self.batch_size)

        states,actions,rewards,new_states,old_log_probs,dones=batch
        states=torch.as_tensor(states,dtype=torch.long)
        actions=torch.as_tensor(actions,dtype=torch.long)
        rewards=torch.as_tensor(rewards,dtype=torch.float32)
        new_states=torch.as_tensor(new_states,dtype=torch.long)
        old_log_probs=torch.as_tensor(old_log_probs,dtype=torch.float32)
        dones=torch.as_tensor(dones,dtype=torch.bool)

        # computr TD target and advantage
        with torch.no_grad():
            state_values=self.critic(states).squeeze(-1)    # [batch]
            new_state_values=self.critic(new_states).squeeze(-1)

            not_dones = (~dones).float()
            td_targets = rewards + self.gamma * not_dones * new_state_values

            advantages=td_targets-state_values
            advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-9)

        new_actor_logits=self.actor(states)
        new_dist=Categorical(logits=new_actor_logits)
        new_log_probs=new_dist.log_prob(actions)

        ratio=torch.exp(new_log_probs-old_log_probs)

        surrogate1=ratio*advantages
        surrogate2=torch.clamp(
            ratio,
            1-self.clip_eps,
            1+self.clip_eps
        )*advantages
        surrogate=torch.min(surrogate1,surrogate2)
        entropy_loss=new_dist.entropy().mean()
        actor_loss=-surrogate.mean()-self.entropy_coef*entropy_loss

        state_values=self.critic(states).squeeze(-1)
        critic_loss=F.mse_loss(state_values,td_targets)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(),critic_loss.item()