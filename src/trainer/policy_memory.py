import random
import torch

from collections import deque


class PolicyMemory:
    def __init__(self, batch_size=40):
        self.buffer = deque(maxlen=10000)
        self.batch_size = batch_size

    def save_into_memory(self, state, action, reward, reward_bl):
        self.buffer.append(
            (
                state,
                action,
                reward,
                reward_bl,
            )
        )

    def sample_from_memory(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, rewards_bl = map(torch.stack, zip(*batch))
        return states, actions, rewards, rewards_bl

    def get_buffer(self):
        return self.buffer

    def get_len(self):
        return len(self.buffer)
