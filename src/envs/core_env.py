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

    def clone(self):
        """Clone this very specific environment and return cloned environment."""
        raise NotImplementedError


class Wrapper(BaseEnv[ObsType, ActType]):
    def __init__(self, env: BaseEnv):
        """
        Wraps the given environment.

        Args:
            env (BaseEnv[ObsType, ActType]): the env to wrap
        """
        self.env = env
        assert isinstance(env, BaseEnv)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action: ActType) -> tuple[ObsType, float, dict[str, Any], bool]:
        return self.env.step(action)

    def reset(self, seed: int | None) -> tuple[ObsType, dict[str, Any]]:
        return self.env.reset(seed)
