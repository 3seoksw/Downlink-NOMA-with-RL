import numpy as np

from envs.core_env import BaseEnv
from users.base_station import BaseStation
from users.noma_user import NOMA_User
from typing import List


class NOMA_Env(BaseEnv):
    """
    ### State Space
    The state is a pair of user and channel.
    For instance, s_t = (U_n, C_k) represents n-th user has been assigned to k-th channel.
    N and K represents the number of users and channels, respectively.
    Hence, the state space contains NK states and the size of it is (NK, 1).
    States are in the form of the following:

           |      U_0       |      U_1       |      U_2       | ... |      U_{N-1}
    _______|________________|________________|________________|_____|__________________
      C_0  |   (U_0, C_0)   |   (U_1, C_0)   |   (U_2, C_0)   | ... |  (U_{N-1}, C_0})
      C_1  |   (U_0, C_1)   |   (U_1, C_1)   |   (U_2, C_1)   | ... |  (U_{N-1}, C_1})
       .   |                |                |                |     |
       .   |                |                |                |     |
       .   |                |                |                |     |
    C_{K-1}| (U_0, C_{K-1}) | (U_1, C_{K-1}) | (U_2, C_{K-1}) | ... | (U_{N-1}, C_{K-1})

    Each state's value is either 0 or 1 denoting false and true, repectively.
    If the state (U_n, C_k) is 1, it means that the n-th user has been assigned to the k-th channel.

    ### Action Space
    The action from this NOMA environment is to assign a channel to a user.
    Here, we suppose the maximum number of users which can be assigned to a channel is 2.
    The action is in the form of `(user_index, channel_index)`.
    """

    def __init__(self, **env_kwargs):
        if env_kwargs is None:
            env_kwargs = {
                "batch_size": 40,
                "num_users": 40,
                "num_channels": 20,
                "channels": [],
                "B_tot": 5,
                "alpha": 2,
                "P_T": 12,
                "seed": 2024,
            }

        self.batch_size = env_kwargs["batch_size"]
        self.N = env_kwargs["num_users"]  # 40
        self.K = env_kwargs["num_channels"]  # 20
        self.bandwidth_total = env_kwargs["B_tot"]  # 5 MHz
        self.channel_bandwidth = self.bandwidth_total / self.K
        self.alpha = env_kwargs["alpha"]  # path loss coefficient (alpha=2)
        self.total_power = env_kwargs["P_T"]  # 2 ~ 12 Watt
        self.seed = env_kwargs["seed"]
        self.channels = env_kwargs["channels"]
        self.noise = env_kwargs["N_0"]  # -170 dBm
        self.channel_variance = (
            self.bandwidth_total * self.noise / self.K
        )  # sigma_{z_k}^2

        self.users = self._generate_noma_users(self.seed)
        self.base_station = self._generate_base_station(self.users)

        self.states = np.zeros((self.N, self.K), dtype=bool)

        # key: channel_idx, value: first assigned user idx to given channel
        self.channel_info = {}

        self.count = 0
        self.done = False

    def _generate_noma_users(self, seed: int | None) -> List[NOMA_User]:
        # TODO: Use seed number to generate random number
        users = []
        for i in range(self.N):
            user = NOMA_User(id=i)
            users.append(user)

        return users

    def _generate_base_station(self, users: List[NOMA_User]) -> BaseStation:
        return BaseStation(users)

    def reset(self, seed: int | None):
        """
        Reset the NOMA environment.

        Returns:
            state: an initial state with the size of NK filled with 0s.
        """
        self.channel_info = {}

        self.count = 0
        self.done = False

        self.states = np.zeros((self.N, self.K), dtype=bool)
        info = {}  # WARN:

        return (self.states, info)

    def step(self, action: tuple):
        """
        Run one step of the NOMA system (environment).

        Args:
            action (ActType): tuple of indices of user and channel.
            `(user_index, channel_index)`

        Returns:
            state: change the state in which the given action is taken to 1.
            Say the user's index as `n` and channel's index as `k`.
            Then states will be updated as `self.states[K * k + n] = 1`.
        """
        self.count += 1

        # States
        user_idx, channel_idx = action
        state_num = self.K * channel_idx + user_idx
        self.states[state_num] = 1

        if self.channel_info.get(channel_idx) is None:
            self.channel_info[channel_idx] = user_idx

        # TODO: Reward
        reward = 0

        # WARN: Info
        info = {}

        # Done
        if self.count == self.N:
            self.done = True

            # TODO: Power allocation with given channel assignment
            power = self._find_optimal_power(channel_idx)
            self.base_station.allocate_resources(user_idx, channel_idx, power)
            self.base_station.multiplex_signals()
            multiplexed_signals = self.base_station.send_multiplexed_signals()

        return (self.states, reward, info, self.done)

    # TODO:
    def derive_data_rate(self, order: int, user_idx, channel_idx):
        power = self.get_power(user_idx, channel_idx)
        cnr = self.get_cnr(user_idx, channel_idx)

        if order == 1:
            sinr = 1 + power * cnr
            data_rate = self.bandwidth_total * np.log2(sinr)
            return data_rate
        elif order == 2:
            power_1 = self.get_power(user)
            sinr = 1 + (power * cnr) / (1 + )

    # TODO:
    def get_power(self, metric: str, user_idx, channel_idx):
        if metric == "MSR":
            self.get_cnr()
            numerator = 
            return 0
        elif metric == "MMR":
            return -1
        else:
            raise KeyError(f"No such metric is available. Choose either `MMR` or `MSR`.")

    def get_cnr(self, channel_idx):
        """
        CNR (channel-to-noise-ratio):
            Gamma^k_n = |h^k_n|^2 / sigma^2_{z_k}
        """
        h = self.get_channel_response(user_idx, channel_idx)
        cnr = np.abs(h) ** 2 / self.channel_variance

        return cnr

    def get_awgn(self):
        """
        Additive White Noise Gaussian Noise (AWGN): z^k_n
        Mean: 0
        Variance: sigma^2_{z^k}
        """

    def get_channel_response(self, user_idx, channel_idx):
        """
        h^k_n = g^k_n * d^{-alpha}_n,
        where `g` follows the Rayleigh distribution,
        and `d` is the distance between n-th user and the base station.
        """
        g = self.sample_from_rayleigh_distribution(user_idx, channel_idx)
        d = self.get_distance_loss(user_idx)
        h = g * d

        return h

    def sample_from_rayleigh_distribution(self, user_idx, channel_idx):
        """
        Rayleight Fading: g^{k}_{n},
        f(x; sigma) = x / sigma^2 * e^{-x^2 / 2 sigma^2}, x >= 0
        """
        x, y = self.users[user_idx].get_location()
        r = np.sqrt(x**2, y**2)

        channel_mean = np.mean(self.channels)
        channel = self.channels[channel_idx]
        var_channel = np.mean((channel - channel_mean) ** 2)

        rng = np.random.default_rng(seed=self.seed)
        rayleigh_dist = rng.rayleigh(var_channel, r)

        return rayleigh_dist

    def get_distance_loss(self, user_idx):
        """Distance Loss: d^{-alpha}_n"""
        x, y = self.users[user_idx].get_location()
        distance = np.sqrt(x**2 + y**2)
        distance_loss = distance ** (-self.alpha)

        return distance_loss
