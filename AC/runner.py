import torch 
import torch.nn.functional as F
from torch.distributions import Categorical
from config import *
from net import *

class DQN_Runner:
    def __init__(self,cfg:config):
        self.cfg=cfg
        device_name=cfg.rl.device
        if device_name=="auto":
            device_name="cuda" if torch.cuda.is_available() else "cpu"
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        self.device=torch.device(device_name)

        self.model=ActorCriticNet(cfg).to(self.device)

        self.optimizer=torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.rl.lr
        )
    
    def select_action(self,state,deterministic:bool=False):
        state=torch.as_tensor(
            state,dtype=torch.float32,device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action_scores,_=self.model(state)

            if deterministic:
                action=action_scores.argmax(dim=-1)
            else:
                distribution=Categorical(logits=action_scores)
                action=distribution.sample()

        return int(action.item())
    
    def train_step(
        self,
        state,
        action,
        reward,
        new_state,
        done
    ):
        state=torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        new_state=torch.as_tensor(
            new_state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        action=torch.as_tensor(
            [action],
            dtype=torch.long,
            device=self.device
        )
        reward=torch.as_tensor(
            [reward],
            dtype=torch.float32,
            device=self.device
        )
        done=torch.as_tensor(
            [done],
            dtype=torch.float32,
            device=self.device
        )

        action_score,state_value=self.model(state)
        # 把原始分数归一化，便于后续损失计算
        distribution=Categorical(logits=action_score)

        # 计算td_target
        with torch.no_grad():
            _,new_state_value=self.model(new_state)
            td_target=(
                reward+self.cfg.rl.gamma*(1-done)*new_state_value
            )

        # 计算优势
        advantage=td_target-state_value
        
        # 分别计算损失
        actor_loss=-(
            distribution.log_prob(action)*advantage.detach()
        ).mean()
        critic_loss=F.mse_loss(state_value,td_target)

        # 熵，鼓励探索
        entropy=-distribution.entropy().mean()

        total_loss=actor_loss+self.cfg.rl.value_coef*critic_loss+self.cfg.rl.entropy_coef*entropy

        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪，避免训练不稳定
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.cfg.rl.max_grad_norm
        )

        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
            "value": state_value.item(),
            "advantage": advantage.item(),
        }
