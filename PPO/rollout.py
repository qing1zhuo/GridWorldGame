from config import GridWorld_Config
from env import GridWorld
from net import Actor,Critic
import numpy as np
from torch.distributions import Categorical
import torch
class GridWorld_Rollout:
    def __init__(self,cfg:GridWorld_Config):
        self.steps_per_env=cfg.steps_per_env
        self.rollout_batch=cfg.rollout_batch
        self.buffer=[]

    def clear(self):
        self.buffer.clear()

    def select_action(self,states:np.ndarray,actor:Actor):
        '''
        参数:
            states numpy数组 [batch]
        返回:
            actions numpy数组 [batch]
            action_log_probs numpy数组 [batch]
        '''
        states_tensor = torch.as_tensor(states, dtype=torch.long)
        with torch.no_grad():
            action_logits = actor(states_tensor)
        dist=Categorical(logits=action_logits)

        actions=dist.sample()
        old_log_probs=dist.log_prob(actions)

        actions=actions.cpu().numpy().astype(np.int64)
        old_log_probs=old_log_probs.cpu().numpy().astype(np.float32)

        return actions,old_log_probs

    def rollout(self,env:GridWorld,actor:Actor):
        """
        进行采样，返回list
        list中每个元素为t时刻的:
            states, actions, rewards, new_states, dones
        """
        # 一次性设置一批次环境 [batch]
        self.clear()
        states=env.reset_batch(self.rollout_batch)
        actions,old_log_probs=self.select_action(states,actor)

        for _ in range(self.steps_per_env):
            rewards,new_states,dones=env.step_batch(states,actions)
            self.buffer.append((states.copy(),actions.copy(),rewards.copy(),new_states.copy(),old_log_probs.copy(),dones.copy()))

            states=new_states
            done_count=np.count_nonzero(dones)
            if done_count:
                states[dones]=env.reset_batch(done_count)
            actions,old_log_probs=self.select_action(states,actor)
        return self.buffer

    def sample(self,batch_size):
        sample_count = self.steps_per_env * self.rollout_batch

        flat_indices = np.random.choice(sample_count, batch_size, replace=False)
        step_indices, env_indices = divmod(flat_indices, self.rollout_batch)
        fields = zip(*self.buffer)
        return tuple(
            np.asarray(field)[step_indices, env_indices]
            for field in fields
        )
