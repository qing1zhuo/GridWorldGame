import numpy

class config():
    class env_config:
        def __init__(self,row_num,col_num,reward,grid):
            self.row_num=row_num
            self.col_num=col_num
            self.reward=reward
            self.grid=numpy.array(grid)
    class rl_config:
        def __init__(self,epsilon,gamma,lr,batch_size,target_update_interval,train_iteration,device,value_coef,entropy_coef,max_grad_norm):
            self.epsilon=epsilon
            self.gamma=gamma
            self.lr=lr
            self.batch_size=batch_size
            self.target_update_interval=target_update_interval
            self.train_iteration=train_iteration
            self.device=device
            self.value_coef=value_coef
            self.entropy_coef=entropy_coef
            self.max_grad_norm=max_grad_norm
    class rollout_config:
        def __init__(self,steps):
            self.steps=steps
    class rand_config:
        def __init__(self,seed):
            self.seed=seed
    class net_config:
        def __init__(self,input_dim,action_dim,hidden_dim):
            self.input_dim=input_dim
            self.action_dim=action_dim
            self.hidden_dim=hidden_dim

    def __init__(
        # env config
        self,
        row_num=5,col_num=5,
        reward={
            "boundary":-10,
            "forbidden":-10,
            "target":1,
            "blank":0
        },
        # 0为空白，1为禁止，2为目标
        grid=[
            [0,0,0,0,0],
            [0,1,1,0,0],
            [0,0,1,0,0],
            [0,1,2,1,0],
            [0,1,0,0,0]
        ],
        # rl config
        epsilon=0.1,
        gamma=0.9,
        lr=1e-3,
        batch_size=32,
        target_update_interval=100,
        train_iteration=1000,
        device="auto",
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        # rollout config
        rollout_steps=10000,
        # random
        seed=123456,
        # net config
        input_dim=2,
        action_dim=5,
        hidden_dim=[20,10]
    ):
        self.env=self.env_config(row_num,col_num,reward,grid)
        self.rl=self.rl_config(
            epsilon,
            gamma,
            lr,
            batch_size,
            target_update_interval,
            train_iteration,
            device,
            value_coef,entropy_coef,
            max_grad_norm
        )
        self.rollout=self.rollout_config(rollout_steps)
        self.rand=self.rand_config(seed=seed)
        self.net=self.net_config(input_dim,action_dim,hidden_dim)
