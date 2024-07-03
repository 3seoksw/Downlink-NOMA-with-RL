import os
import torch

from envs.noma_env import NOMA_Env
from tqdm import tqdm


class NOMA_Searcher:
    def __init__(
        self,
        num_users: int = 6,
        num_channels: int = 3,
        P_T: int = 12,
        seed: float = 2024,
    ):
        self.N = num_users
        self.K = num_channels
        self.seed = seed
        self.env = NOMA_Env(num_users=self.N, num_channels=self.K, P_T=P_T)

    def set_NK(self, num_users: int, num_channels):
        self.N = num_users
        self.K = num_channels
        assert self.K * 2 == self.N

    def set_seed(self, seed: float):
        self.seed = seed

    def _generate_all_pairs(self, users: list):
        if len(users) < 2:
            return [[]]

        pairs = []
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                pair = (users[i], users[j])
                remaining_users = users[:i] + users[i + 1 : j] + users[j + 1 :]
                for rest in self._generate_all_pairs(remaining_users):
                    pairs.append([pair] + rest)

        return pairs

    def _create_action_sets(self, pairs: list):
        action_sets = []
        for pair in pairs:
            actions = []

            for k in range(len(pair)):
                usr0, usr1 = pair[k]
                action0 = k * self.N + usr0
                action1 = k * self.N + usr1
                actions.append(action0)
                actions.append(action1)
            action_sets.append(actions)

        return action_sets

    def exhaustive_search(self):
        users = list(range(self.N))
        pairs = self._generate_all_pairs(users)
        action_sets = self._create_action_sets(pairs)

        total = 0
        max_reward = 0
        min_reward = 1000
        action_sets = tqdm(action_sets)
        for actions in action_sets:
            state, info = self.env.reset(self.seed)
            reward = 0
            for action in actions:
                action = torch.tensor([action])
                state, reward, info, done = self.env.step(action)

            if reward == 0:
                raise KeyError()

            max_reward = max(reward, max_reward)
            min_reward = min(reward, min_reward)
            total += reward

        avg_reward = total / len(action_sets)

        return min_reward, avg_reward, max_reward
