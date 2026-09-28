class GridWorld_Config:
    def __init__(
        self,
        # random cfg
        rand_seed=123456,
        # env cfg
        row_num=5,
        col_num=5,
        rewards={
            "blank":0,
            "boundary":-10,
            "forbidden":-10,
            "target":10
        },
        grid=[
            [0,0,0,0,0],
            [0,1,1,0,0],
            [0,0,1,0,0],
            [0,1,2,1,0],
            [0,1,0,0,0]
        ],
        action2dx={0:-1 , 1:0 , 2:1 , 3:0 , 4:0},
        action2dy={0:0 , 1:1 , 2:0 , 3:-1 ,4:0},
        idx2grid={0:"blank",1:"forbidden",2:"target"},
        # actor_net config
        input_dim=25,
        action_dim=5,
        actor_hidden_dim=[64,64],
        # critic_net config
        critic_hidden_dim=[64,64],
        # rollout config
        steps_per_env=32,   # 一个环境走几步
        rollout_batch=100,  # 一次开多少个环境
        # runner config
        gamma=0.95,
        clip_eps=0.2,
        entropy_coef_start=0.01,    # 其实熵权重
        entropy_coef_end=0.001,     # 末尾熵权重
        batch_size=256,     # 训练一轮用多少条数据
        actor_lr=3e-4,
        critic_lr=3e-4,
        # train config
        train_iterations=1500,
        update_per_iter=5
    ):
        self.rand_seed=rand_seed

        self.row_num=row_num
        self.col_num=col_num
        self.rewards=rewards
        self.grid=grid
        self.action2dx=action2dx
        self.action2dy=action2dy
        self.idx2grid=idx2grid

        self.input_dim=input_dim
        self.action_dim=action_dim
        self.actor_hidden_dim=actor_hidden_dim

        self.critic_hidden_dim=critic_hidden_dim

        self.steps_per_env=steps_per_env
        self.rollout_batch=rollout_batch

        self.gamma=gamma
        self.clip_eps=clip_eps
        self.entropy_coef_start=entropy_coef_start
        self.entropy_coef_end=entropy_coef_end
        self.batch_size=batch_size
        self.actor_lr=actor_lr
        self.critic_lr=critic_lr

        self.train_iterations=train_iterations
        self.update_per_iter=update_per_iter