import random
from collections import deque
from config import *
from env import *
import numpy as np
import torch
def encode_state(state,cfg:config):
    return np.array(
        [
            state[0]/(cfg.env.row_num-1),
            state[1]/(cfg.env.col_num-1)
        ],
        dtype=np.float32
    )

class ReplayBuffer:
    def __init__(self,capacity=10000,seed=None):
        self.buffer=deque(maxlen=capacity)
        self.rng=random.Random(seed)
    
    # 将一步的结果存入，这里state要求归一化
    def push(self,state,action,reward,new_state,done):
        self.buffer.append(
            (
                np.asarray(state,dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(new_state,dtype=np.float32),
                float(done)
            )
        )

    # 取出一批数据
    def sample(self,batch_size,device="cpu"):
        if batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size={batch_size} exceeds buffer size={len(self.buffer)}"
            )
        batch=self.rng.sample(self.buffer,batch_size)
        states,actions,rewards,new_states,dones=zip(*batch)

        return (
            torch.as_tensor(np.asarray(states),dtype=torch.float32,device=device),
            torch.as_tensor(actions,dtype=torch.long,device=device),
            torch.as_tensor(rewards,dtype=torch.float32,device=device),
            torch.as_tensor(np.asarray(new_states),dtype=torch.float32,device=device),
            torch.as_tensor(dones,dtype=torch.float32,device=device)
        )
    def __len__(self):
        return len(self.buffer)

def uniform_rollout(
    env:GridWorld,
    cfg:config,
    replay_buffer
):
    rng=np.random.default_rng(cfg.rand.seed)

    state=env.reset()
    episode_return=0.0
    rewards=[]

    for _ in range(cfg.rollout.steps):
        action = int(rng.integers(0, cfg.net.output_dim))

        new_state,reward,done=env.step(
            action=action
        )

        replay_buffer.push(
            state=encode_state(state,cfg),
            action=action,
            reward=reward,
            new_state=encode_state(new_state,cfg),
            done=done
        )

        episode_return+=reward
        rewards.append(reward)
        state=new_state

        if done:
            state=env.reset()
    return episode_return,rewards
