import numpy as np


def corrupt_state(state, error_prob: float):
    """
    Assumes channel state information (CSI) is imperfect.

    input: non-batch state
    """
    state_size = state.shape[0]

    corrupts = []
    for s in range(state_size):
        if np.random.random() <= error_prob:
            corrupts.append(s)

    for c in corrupts:
        state[c, 0] = 0
        state[c, 1] = 0

    return state
