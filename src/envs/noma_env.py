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
                "noise": -170,
                "min_data_rate": 6,
                "metric": "MSR",
            }

        self.batch_size = env_kwargs["batch_size"]  # 40
        self.N = env_kwargs["num_users"]  # 40
        self.K = env_kwargs["num_channels"]  # 20
        self.bandwidth_total = env_kwargs["B_tot"]  # 5 MHz
        self.channel_bandwidth = self.bandwidth_total / self.K  # 2,500,000 Hz
        self.alpha = env_kwargs["alpha"]  # path loss coefficient (alpha=2)
        self.total_power = env_kwargs["P_T"]  # 2 ~ 12 Watt
        self.seed = env_kwargs["seed"]  # 2024
        self.channels = env_kwargs["channels"]
        self.noise = env_kwargs["N_0"]  # -170 dBm
        self.channel_variance = (  # WARN: erase `/ self.K` if needed
            self.bandwidth_total * self.noise / self.K
        )  # sigma_{z_k}^2
        self.min_data_rate = env_kwargs["min_data_rate"]
        self.metric = env_kwargs["metric"]

        self.states = np.zeros((self.K * self.N), dtype=bool)
        self.info = {"n_steps": 0}  # TODO: which keys to be inserted
        self.done = False

        # key: channel_idx, value: list[user_idx, user_idx2]
        self.channel_info = {}

    def reset(self, seed: int | None):
        """
        Reset the NOMA environment.

        Returns:
            state: an initial state with the size of NK filled with 0s.
        """
        self.channel_info = {}

        self.done = False

        self.states = np.zeros((self.N * self.K), dtype=bool)
        self.info = {"n_steps": 0}

        return (self.states, self.info)

    def step(self, action: tuple):
        """
        Run one step of the NOMA system (environment).

        Args:
            action (ActType): tuple of indices of user and channel.
            `(user_index, channel_index)`

        Returns:
            state: change the state in which the given action is taken to 1.
            Say the user's index as `n` and channel's index as `k`.
            Then states will be updated as `self.states[N * k + n] = 1`.
        """
        # States
        user_idx, channel_idx = action
        state_num = self.K * channel_idx + user_idx
        self.states[state_num] = 1

        if self.channel_info.get(channel_idx) is None:
            # First assigned
            self.channel_info[channel_idx] = list(user_idx)
        else:
            # Secondly assigned
            self.channel_info[channel_idx].append(user_idx)

        # WARN:
        self.info["n_steps"] += 1

        # Done
        if self.info["n_steps"] == self.N:
            self.done = True

            # TODO: Power allocation with given channel assignment
            self.on_epoch_end_conduct_allocation()

        return (self.states, reward, self.info, self.done)

    # FIXME: erase `base_station`
    def on_epoch_end_conduct_allocation(self):
        """Allocate channel and power to all users"""
        for channel_idx in range(self.K):
            channel_info = self.channel_info[channel_idx]
            if len(channel_info) == 0:
                continue
            elif len(channel_info) == 1:
                user_idx = channel_info[0]
                power = self.get_power(user_idx, channel_info, self.metric)
                self.base_station.allocate_resources(user_idx, channel_idx, power)
            else:  # len(channel_info) == 2
                user_indices = channel_info
                for user_idx in user_indices:
                    power = self.get_power(user_idx, channel_idx, self.metric)
                    self.base_station.allocate_resources(user_idx, channel_idx, power)

    def get_data_rate(self, channel_idx, n, metric):
        power_0 = self.get_power(channel_idx, 0, metric)
        cnr_0 = self.get_cnr(channel_idx, 0)
        channel = self.channel_bandwidth * channel_idx

        if n == 0:
            return channel * np.log2(1 + power_0 * cnr_0)
        else:  # n == 1
            power_1 = self.get_power(channel_idx, 1, metric)
            cnr_1 = self.get_cnr(channel_idx, 1)
            return channel * np.log2(1 + (power_1 * cnr_1) / (1 + power_0 * cnr_1))

    # TODO:
    def get_power(self, channel_idx, n: int = 0, metric: str = "MSR"):
        user_idx = self.channel_info[channel_idx][n]
        q = self.get_power_budget(channel_idx)
        if metric == "MSR":
            gamma_1 = self.get_cnr(channel_idx, 1)
            A = self.get_A(channel_idx)
            numerator = 

            p_0 = (gamma_1 * q - A + 1) / (A * gamma_1)
            p_1 = q - p_0

            return (p_0, p_1)
        elif metric == "MMR":
            return -1
        else:
            raise KeyError(f"No such metric is available. Choose either `MMR` or `MSR`.")

    # TODO
    def get_power_budget(self, channel_idx):
        """Calculate q^k"""
        l = 0
        q = []
        for channel_idx in range(self.K):
            q_k = 
            q[channel_idx] = q_k

    def get_sinr(self, channel_idx):
        cnr_0 = self.get_cnr(channel_idx, 0)
        cnr_1 = self.get_cnr(channel_idx, 1)

        A = self.get_A(channel_idx)

        return (A * (A  - 1)) / cnr_0 + (A - 1) / cnr_1

    # WARN: A^k >= 2
    def get_A(self, channel_idx):
        channel_bandwidth = channel_idx * self.channel_bandwidth
        return 2 ** (self.min_data_rate / channel_bandwidth)

    def get_cnr(self, channel_idx, n: int = 0):
        """
        CNR (channel-to-noise-ratio):
            Gamma^k_n = |h^k_n|^2 / sigma^2_{z_k}
        """
        user_idx = self.channel_info[channel_idx][n]
        h = self.get_channel_response(user_idx, channel_idx)
        cnr = np.abs(h) ** 2 / self.channel_variance

        return cnr

    def get_awgn(self):
        """
        Additive White Noise Gaussian Noise (AWGN): z^k_n
        Mean: 0
        Variance: sigma^2_{z^k}
        """

    def get_channel_response(self, channel_idx, n: int = 0):
        """
        h^k_n = g^k_n * d^{-alpha}_n,
        where `g` follows the Rayleigh distribution,
        and `d` is the distance between n-th user and the base station.

        n is either 0 or 1.
        """
        user_idx = self.channel_info[channel_idx][n]
        g = self.sample_from_rayleigh_distribution(channel_idx, n)
        d = self.get_distance_loss(user_idx)
        h = g * d

        return h

    def sample_from_rayleigh_distribution(self, channel_idx, n: int = 0):
        """
        Rayleight Fading: g^{k}_{n},
        f(x; sigma) = x / sigma^2 * e^{-x^2 / 2 sigma^2}, x >= 0

        Args:
            channel_idx: k
            `n` is either 0 or 1 denoting whether the user has been assigned first
            or not.
        """
        user_idx = self.channel_info[channel_idx][n]
        x, y = self.users[user_idx].get_location()
        r = np.sqrt(x**2, y**2)

        # WARN: Deprecated
        # channel_mean = np.mean(self.channels)
        # channel = self.channels[channel_idx]
        # var_channel = np.mean((channel - channel_mean) ** 2)

        rng = np.random.default_rng(seed=self.seed)
        rayleigh_dist = rng.rayleigh(self.channel_variance, r)

        return rayleigh_dist

    def get_distance_loss(self, user_idx):
        """Distance Loss: d^{-alpha}_n"""
        x, y = self.users[user_idx].get_location()
        distance = np.sqrt(x**2 + y**2)
        distance_loss = distance ** (-self.alpha)

        return distance_loss
