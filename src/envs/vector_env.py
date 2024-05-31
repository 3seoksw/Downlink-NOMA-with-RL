import torch

from envs.core_env import BaseEnv, Wrapper
from typing import override, Optional


VecType = torch.Tensor


class VectorEnv(Wrapper):
    def __init__(self, env: BaseEnv, num_envs: int):
        super().__init__(env)
        self.envs = [env.clone() for _ in range(num_envs)]
        self.num_envs = num_envs
        print(f"VectorEnv: PyTorch device, {self.device} loaded")

    @override
    def reset(self, seed: Optional[int] = None) -> tuple[VecType, list]:
        """Returns multiple reset data with the number of `num_envs`."""
        raise NotImplementedError

    @override
    def step(
        self, actions: tuple[VecType, VecType]
    ) -> tuple[VecType, VecType, list, VecType]:
        """
        Returns multiple `state`, `reward`, `info`, and `done` data
        with the number of `num_envs`.
        """
        raise NotImplementedError


class VectorizedEnv(VectorEnv):
    def __init__(self, env: BaseEnv, num_envs: int):
        super().__init__(env, num_envs)

    def reset(self, seed: Optional[int] = None):
        states = []
        infos = []
        for env in self.envs:
            state, info = env.reset(seed)
            states.append(state.clone().detach().to(self.device))
            infos.append(info)

        states = torch.stack(states)

        return states, infos

    def step(self, actions):
        states = []
        rewards = []
        infos = []
        dones = []
        for env, action in zip(self.envs, actions):
            state, reward, info, done = env.step(action)
            states.append(state.clone().detach().to(self.device))
            rewards.append(torch.tensor(reward).to(self.device))
            infos.append(info)
            dones.append(torch.tensor(done, dtype=torch.bool).to(self.device))

        states = torch.stack(states)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        return (states, rewards, infos, dones)
