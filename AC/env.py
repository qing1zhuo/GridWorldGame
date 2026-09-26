import numpy as np


class GridWorld:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.rand.seed)
        self.state = None

    def reset(self):
        row = int(self.rng.integers(0, self.cfg.env.row_num))
        col = int(self.rng.integers(0, self.cfg.env.col_num))

        self.state = (row, col)
        return self.state

    def step(self, action):
        # action: up0,right1,down2,left3,stay4
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        new_x,new_y=self.state
        done=False

        if action==0:
            new_x-=1
        elif action==1:
            new_y+=1
        elif action==2:
            new_x+=1
        elif action==3:
            new_y-=1
        elif action==4:
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

        if new_x<0 or new_x>=self.cfg.env.row_num or new_y<0 or new_y>=self.cfg.env.col_num:
            reward=self.cfg.env.reward["boundary"]
            done=False
            return self.state,reward,done

        if self.cfg.env.grid[new_x,new_y]==1:
            reward=self.cfg.env.reward["forbidden"]
            done=False
            self.state=(new_x,new_y)
            return self.state,reward,done

        if self.cfg.env.grid[new_x,new_y]==2:
            reward=self.cfg.env.reward["target"]
            done=True
            self.state=(new_x,new_y)
            return self.state,reward,done

        else:
            reward=self.cfg.env.reward["blank"]
            done=False
            self.state=(new_x,new_y)
            return self.state,reward,done
