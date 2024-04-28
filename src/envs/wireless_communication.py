from envs.core_env import BaseEnv, Wrapper
from envs.core_env import ObsType, ActType, WrappedObsType, WrappedActType


class WirelessCommunication:
    def __init__(self, tx, rx):
        self.tx = tx
        self.rx = rx

    def step(self, action: ActType):
        """"""
        selected_channel = action
        # calculate power with channel
        self.tx.allocate_resources(selected_channel, power)

    def reset(self):
        """
        Reset the environment to an initial state.

        Args:
            seed (optional: int)

        Returns:
            observation (ObsType)
            info (dict[str, Any])
        """
        return ObsType, dict[str, float]

    def get_tx(self):
        return self.tx

    def get_rx(self):
        return self.rx
