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
