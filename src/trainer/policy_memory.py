import random
import torch

from collections import deque


class PolicyMemory:
    def __init__(self, batch_size=40, use_experience_replay=True):
        self.buffer = deque(maxlen=10000)
        self.batch_size = batch_size
        self.use_experience_replay = use_experience_replay

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
        if self.use_experience_replay:
            if len(self.buffer) < self.batch_size:
                raise ValueError()
            batch = []
            for _ in range(self.batch_size):
                batch.append(self.buffer.popleft())
        else:
            batch = random.sample(self.buffer, self.batch_size)

        states, actions, rewards, rewards_bl = map(torch.stack, zip(*batch))
        return states, actions, rewards, rewards_bl

    def get_buffer(self):
        return self.buffer

    def get_len(self):
        return len(self.buffer)
