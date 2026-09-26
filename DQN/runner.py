import torch 
import torch.nn.functional as F
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

        self.main_net=main_net(
            cfg.net.input_dim,cfg.net.output_dim,cfg.net.hidden_dim
        ).to(self.device)
        self.target_net=target_net(
            cfg.net.input_dim,cfg.net.output_dim,cfg.net.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()
        for parameter in self.target_net.parameters():
            parameter.requires_grad_(False)

        self.update_count=0

        self.optimizer=torch.optim.Adam(
            self.main_net.parameters(),
            cfg.rl.lr
        )
    
    def select_action(self,state):
        state=torch.as_tensor(
            state,dtype=torch.float32,device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values=self.main_net(state)

        return int(q_values.argmax(dim=-1).item())

    def train_step(self,replay_buffer):
        states,actions,rewards,new_states,dones=replay_buffer.sample(
            batch_size=self.cfg.rl.batch_size,
            device=self.device,
        )

        # 用当前状态和主网络预测当前q值
        predicted_q=self.main_net(states).gather(dim=1,index=actions.unsqueeze(1)).squeeze(1)
        # 利用价值网络计算下一状态q值，并乘上衰减系数
        with torch.no_grad():
            next_q=self.target_net(new_states).max(dim=1).values
            target_q=(rewards+self.cfg.rl.gamma*(1-dones)*next_q)

        loss=F.mse_loss(predicted_q,target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count+=1

        if self.update_count%self.cfg.rl.target_update_interval==0:
            self.target_net.load_state_dict(self.main_net.state_dict())

        return loss.item()
