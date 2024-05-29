import random
import torch

from collections import deque


class ReplayMemory:
    def __init__(self, batch_size):
        self.buffer = deque(maxlen=10000)
        self.batch_size = batch_size

    def save_into_memory(self, prev_state, state, action, reward):
        prev_state = prev_state.squeeze(0)
        state = state.squeeze(0)
        
        self.buffer.append((
            prev_state,
            state,
            action,
            reward,
        ))

    def sample_from_memory(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, next_state, action, reward = map(torch.stack, zip(*batch))
        return state, next_state, action, reward

    def get_buffer(self):
        return self.buffer

    def get_len(self):
        return len(self.buffer)
