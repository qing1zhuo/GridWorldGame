"""One-hot features avoid interference between neighboring grid cells."""
import numpy as np

def encode_state(state, cfg):
    features = np.zeros(cfg.net.input_dim, dtype=np.float32)
    features[state[0]*cfg.env.col_num+state[1]] = 1.0
    return features
