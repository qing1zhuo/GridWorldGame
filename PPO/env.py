from config import GridWorld_Config
import numpy as np

class GridWorld:
    def __init__(self,cfg:GridWorld_Config):
        self.row_num=cfg.row_num
        self.col_num=cfg.col_num
        self.rewards=cfg.rewards
        self.grid=cfg.grid
        self.rng=np.random.default_rng(cfg.rand_seed)
        self.action2dx=np.asarray(
            [cfg.action2dx[i] for i in range(len(cfg.action2dx))],
            dtype=np.int64,
        )
        self.action2dy=np.asarray(
            [cfg.action2dy[i] for i in range(len(cfg.action2dy))],
            dtype=np.int64,
        )
        self.idx2grid=cfg.idx2grid
        self.nonterminal_states=np.asarray(
            [
                state
                for state,cell in enumerate(np.asarray(self.grid).ravel())
                if self.idx2grid[cell]!="target"
            ],
            dtype=np.int64,
        )

    def step_batch(self,states:np.ndarray,actions:np.ndarray):
        """
        参数和返回均使用numpy的ndarray
        批量处理batch的states和actions
        states: [batch]
        actions: [batch]
        返回:
        rewards: [batch]
        new_states: [batch]
        dones: [batch]
        """
        cur_x=states//self.col_num   # [batch]
        cur_y=states%self.col_num    # [batch]
        new_x=cur_x+self.action2dx[actions]      # [batch]
        new_y=cur_y+self.action2dy[actions]      # [batch]

        rewards=[]
        new_states=[]
        dones=[]

        batch_size=len(states)
        for i in range(batch_size):
            # 出边界特别判断
            if new_x[i]<0 or new_x[i]>=self.row_num or new_y[i]<0 or new_y[i]>=self.col_num:
                rewards.append(self.rewards["boundary"])
                new_states.append(cur_x[i]*self.col_num+cur_y[i])
                dones.append(False)
                continue
            # 计算新位置的回报
            new_state_name=self.idx2grid[self.grid[new_x[i]][new_y[i]]]
            rewards.append(self.rewards[new_state_name])
            new_states.append(new_x[i]*self.col_num+new_y[i])
            if new_state_name=="target":
                dones.append(True)
            else:
                dones.append(False)


        rewards=np.asarray(rewards,dtype=np.float32)
        new_states=np.asarray(new_states,dtype=np.long)
        dones=np.asarray(dones,dtype=np.bool)
        return rewards,new_states,dones

    def reset_batch(self,batch_size)->np.ndarray:
        """
        参数: batch_size 一个整数
        返回: states 状态列表 np数组 [batch]
        """
        states=self.rng.choice(self.nonterminal_states,size=batch_size)
        return states
    def reset_single(self):
        """
        参数: batch_size 一个整数
        返回: state 状态 一个整数
        """
        state=self.rng.choice(self.nonterminal_states)
        return state
