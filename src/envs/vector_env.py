import numpy as np

from envs.core_env import BaseEnv, Wrapper
from typing import override


VecType = np.ndarray


class VectorEnv(Wrapper):
    def __init__(self, env: BaseEnv, num_envs: int):
        self.envs = [env.clone() for _ in range(num_envs)]
        self.num_envs = num_envs

    @override
    def reset(self, seed: int | None) -> tuple[VecType, VecType]:
        """Returns multiple reset data with the number of `num_envs`."""
        raise NotImplementedError

    @override
    def step(self, actions: np.ndarray) -> tuple[VecType, VecType, VecType, VecType]:
        """
        Returns multiple `state`, `reward`, `info`, and `done` data
        with the number of `num_envs`.
        """
        raise NotImplementedError


class VectorizedEnv(VectorEnv):
    def __init__(self, env: BaseEnv, num_envs: int = 2):
        super().__init__(env, num_envs)

    def reset(self, seed: int | None):
        return tuple([env.reset(seed) for env in self.envs])

    def step(self, actions):
        states = np.array([])
        rewards = np.array([])
        infos = np.array([])
        dones = np.array([])
        for env, action in zip(self.envs, actions):
            state, reward, info, done = env.step(action)
            np.append(states, state)
            np.append(rewards, reward)
            np.append(infos, info)
            np.append(dones, done)

        return (states, rewards, infos, dones)
