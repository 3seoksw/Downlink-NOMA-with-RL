from typing import Generic, TypeVar, Any


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class BaseEnv(Generic[ObsType, ActType]):
    """
    Base environment.
    """

    def step(self, action: ActType) -> tuple[ObsType, float, dict[str, Any], bool]:
        """
        Run one time-step of the environment.

        Args:
            action: an action provided by the agent to update the env state.

        Returns:
            observation (ObsType)
            reward (float)
            info (dict[str, Any])
            done (bool)
        """
        raise NotImplementedError

    def reset(self, seed: int | None) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed (optional: int)

        Returns:
            observation (ObsType)
            info (dict[str, Any])
        """
        raise NotImplementedError


WrappedObsType = TypeVar("WrappedObsType")
WrappedActType = TypeVar("WrappedActType")


class Wrapper(
    BaseEnv[WrappedObsType, WrappedActType],
    Generic[WrappedObsType, WrappedActType, ObsType, ActType],
):
    def __init__(self, env: BaseEnv[ObsType, ActType]):
        """
        Wraps the given environment.

        Args:
            env (BaseEnv[ObsType, ActType]): the env to wrap
        """
        self.env = env
        assert isinstance(env, BaseEnv)

    def step(
        self, action: WrappedActType
    ) -> tuple[WrappedObsType, float, dict[str, Any], bool]:
        return self.env.step(action)
