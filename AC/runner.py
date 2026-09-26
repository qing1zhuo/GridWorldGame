import torch
from torch.distributions import Categorical
from net import ActorCriticNet

class ActorCriticRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        name = cfg.rl.device
        if name=='auto':
            name='cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(name)
        self.model = ActorCriticNet(cfg).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.model.actor_head.parameters(),lr=cfg.rl.lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critic_head.parameters(),lr=cfg.rl.critic_lr)
        self.updates = 0

    def select_action(self, state, deterministic=False):
        state = torch.as_tensor(state,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            logits,_ = self.model(state)
            action = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
        return int(action.item())

    def train_batch(self, states, actions, rewards, next_states, dones):
        """Sampled one-step AC. No optimal values/actions are supplied here."""
        self.updates += 1
        logits,values = self.model(states)
        distribution = Categorical(logits=logits)
        with torch.no_grad():
            _,next_values = self.model(next_states)
            targets = rewards+self.cfg.rl.gamma*(~dones)*next_values
        advantage = (targets-values)/self.cfg.reward_scale
        actor_loss = -(distribution.log_prob(actions)*advantage.detach()).mean()
        critic_loss = advantage.square().mean()
        entropy = distribution.entropy().mean()
        entropy_coef = self.cfg.rl.entropy_coef*max(0.0,1-self.updates/self.cfg.rl.entropy_decay_steps)
        decay = max(self.cfg.rl.min_lr_ratio,1-self.updates/self.cfg.rl.lr_decay_steps)
        for group in self.actor_optimizer.param_groups:
            group['lr']=self.cfg.rl.lr*decay
        for group in self.critic_optimizer.param_groups:
            group['lr']=self.cfg.rl.critic_lr*decay
        self.actor_optimizer.zero_grad()
        (actor_loss-entropy_coef*entropy).backward()
        torch.nn.utils.clip_grad_norm_(self.model.actor_head.parameters(),self.cfg.rl.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic_head.parameters(),self.cfg.rl.max_grad_norm)
        self.critic_optimizer.step()
        return dict(loss=float(actor_loss.detach()+critic_loss.detach()-entropy_coef*entropy.detach()),
            actor_loss=actor_loss.item(),critic_loss=critic_loss.item(),entropy=entropy.item(),
            entropy_coef=entropy_coef,sampled_mean_reward=rewards.mean().item())

DQN_Runner = ActorCriticRunner  # backward-compatible import
