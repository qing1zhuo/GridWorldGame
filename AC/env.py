"""Penalty cells are traversable; targets terminate with zero future reward."""
import numpy as np

class GridWorld:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.rand.seed)
        self.nonterminal_ids = np.flatnonzero(cfg.env.grid.ravel() != 2)
        self.state = None

    def reset(self, state=None):
        if state is None:
            state = divmod(int(self.rng.choice(self.nonterminal_ids)), self.cfg.env.col_num)
        row, col = state
        if not (0 <= row < self.cfg.env.row_num and 0 <= col < self.cfg.env.col_num):
            raise ValueError("start state out of bounds")
        if self.cfg.env.grid[row,col] == 2:
            raise ValueError("a terminal target cannot be an episode start")
        self.state = (int(row), int(col))
        return self.state

    def transition_batch(self, state_ids, actions):
        """Pure simulator. Training supplies only freshly sampled Actor actions.

        Targets are absorbing with zero reward when queried by the evaluator.
        """
        ids, actions = np.broadcast_arrays(np.asarray(state_ids,dtype=np.int64), np.asarray(actions,dtype=np.int64))
        rows, cols = self.cfg.env.grid.shape
        if np.any((ids<0)|(ids>=rows*cols)) or np.any((actions<0)|(actions>=5)):
            raise ValueError("invalid state id or action")
        row, col = ids//cols, ids%cols
        nr = row + np.asarray([-1,0,1,0,0])[actions]
        nc = col + np.asarray([0,1,0,-1,0])[actions]
        boundary = (nr<0)|(nr>=rows)|(nc<0)|(nc>=cols)
        next_ids = np.where(boundary, ids, np.clip(nr,0,rows-1)*cols+np.clip(nc,0,cols-1))
        cells = self.cfg.env.grid.ravel()[next_ids]
        rewards = np.where(cells==2, self.cfg.env.reward['target'],
            np.where(cells==1, self.cfg.env.reward['forbidden'],self.cfg.env.reward['blank'])).astype(np.float64)
        rewards = np.where(boundary,self.cfg.env.reward['boundary'],rewards)
        terminal = self.cfg.env.grid.ravel()[ids]==2
        done = terminal | ((cells==2)&~boundary)
        return np.where(terminal,ids,next_ids), np.where(terminal,0.0,rewards), done

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if self.cfg.env.grid[self.state]==2:
            raise RuntimeError("Episode terminated; call reset().")
        sid = self.state[0]*self.cfg.env.col_num+self.state[1]
        ns,reward,done = self.transition_batch([sid],[action])
        self.state = divmod(int(ns[0]),self.cfg.env.col_num)
        return self.state,float(reward[0]),bool(done[0])
