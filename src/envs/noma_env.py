import torch
import numpy as np

from envs.core_env import BaseEnv


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
        default_env_kwargs = {
            "batch_size": 40,
            "num_users": 40,
            "num_channels": 20,
            "channels": [],
            "B_tot": 5,
            "alpha": 2,
            "P_T": 12,
            "seed": 2024,
            "N_0": -170,
            "min_data_rate": 6,
            "metric": "MSR",
        }
        self.env_kwargs = {**default_env_kwargs, **env_kwargs}

        self.batch_size = self.env_kwargs["batch_size"]  # 40
        self.N = self.env_kwargs["num_users"]  # 40
        self.K = self.env_kwargs["num_channels"]  # 20
        self.bandwidth_total = self.env_kwargs["B_tot"]  # 5 MHz
        self.channel_bandwidth = self.bandwidth_total / self.K  # 2,500,000 Hz
        self.alpha = self.env_kwargs["alpha"]  # path loss coefficient (alpha=2)
        self.total_power = self.env_kwargs["P_T"]  # 2 ~ 12 Watt
        self.seed = self.env_kwargs["seed"]  # 2024
        self.channels = self.env_kwargs["channels"]
        self.noise = self.env_kwargs["N_0"]  # -170 dBm
        self.channel_variance = (  # WARN: erase `/ self.K` if needed
            self.bandwidth_total * self.noise / self.K
        )  # sigma_{z_k}^2
        self.min_data_rate = self.env_kwargs["min_data_rate"]  # 2 bps/Hz
        self.metric = self.env_kwargs["metric"]

        self.states = np.zeros((self.K * self.N), dtype=float)
        self.info = {"n_steps": 0}  # TODO: which keys to be inserted
        self.done = False

        # key: channel_idx, value: list[(user_idx0, cnr0), (user_idx1, cnr1)]
        self.channel_info = {}

        # NOTE: See `_generate_user()` for more information
        self.user_info = []

    def reset(self, seed: int | None):
        """
        Reset the NOMA environment.

        Returns:
            state: an initial state with the size of NK filled with 0s.
        """
        for i in range(self.N):
            user_dict = self._generate_user(i, seed)
            self.user_info.append(user_dict)

        self.channel_info = {}

        self.done = False

        self.states = np.zeros((self.N, 2))
        self.states = torch.tensor(self.states, dtype=torch.float32)

        self.user_channel_indices = np.negative(np.ones((self.N, 2)))
        self.user_channel_indices = torch.tensor(
            self.user_channel_indices, dtype=torch.int
        )

        # self.info = {"n_steps": 0, "user_channel_indices": self.user_channel_indices}
        self._update_info(0, self.user_channel_indices)

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
        step = self.info["n_steps"]

        user_idx, channel_idx = action
        self.allocate_resources(channel_idx, user_idx)

        # States update
        self.states[step][0] = self.user_info[user_idx]["distance"]

        for usr_cnr_pair in self.channel_info[channel_idx]:
            usr, cnr = usr_cnr_pair
            if user_idx == usr:
                self.states[step][1] = cnr

        # User-channel pair update
        self.user_channel_indices[step][0] = user_idx
        self.user_channel_indices[step][1] = channel_idx
        self._update_info(step + 1, self.user_channel_indices)

        reward = self.user_info[user_idx]["data_rate"]

        if self.info["n_steps"] == self.N:
            self.done = True

        return (self.states, reward, self.info, self.done)

    def clone(self):
        return NOMA_Env(env_kwargs=self.env_kwargs)

    def allocate_resources(self, channel_idx, user_idx):
        if self.channel_info.get(channel_idx) is None:
            power = self.get_power(channel_idx, 0, self.metric)
            self.set_power(user_idx, power)

            data_rate = self.get_data_rate(channel_idx, 0, self.metric)
            self.set_data_rate(user_idx, data_rate)

            cnr = self.get_cnr(channel_idx, 0)
            self.channel_info[channel_idx] = list((user_idx, cnr))
        else:
            power = self.get_power(channel_idx, 1, self.metric)
            self.set_power(user_idx, power)

            data_rate = self.get_data_rate(channel_idx, 1, self.metric)
            self.set_data_rate(user_idx, data_rate)

            cnr = self.get_cnr(channel_idx, 1)
            self.channel_info[channel_idx].append((user_idx, cnr))

    def set_data_rate(self, user_idx, data_rate):
        self.user_info[user_idx]["data_rate"] = data_rate

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

    def set_power(self, user_idx, power):
        self.user_info[user_idx]["power"] = power

    # TODO:
    def get_power(self, channel_idx, n: int = 0, metric: str = "MSR"):
        user_idx = self.channel_info[channel_idx][n]
        q = self.get_power_budget(channel_idx, metric)
        if metric == "MSR":
            gamma_1 = self.get_cnr(channel_idx, 1)
            A = self.get_A(channel_idx)

            p_0 = (gamma_1 * q - A + 1) / (A * gamma_1)
            p_1 = q - p_0

            return (p_0, p_1)
        elif metric == "MMR":
            return -1
        else:
            raise KeyError(
                f"No such metric is available. Choose either `MMR` or `MSR`."
            )

    # TODO
    def get_power_budget(self, channel_idx, metric: str):
        """Calculate q^k"""
        if metric == "MSR":
            q = 0
        elif metric == "MMR":
            q = 0
        else:
            raise KeyError("Choose either `MSR` or `MMR`.")

        return q

    def get_gamma_k(self, channel_idx):
        A = self.get_A(channel_idx)
        cnr_0 = self.get_cnr(channel_idx, 0)
        cnr_1 = self.get_cnr(channel_idx, 1)

        gamma_k = (A * (A - 1)) / cnr_0 + (A - 1) / cnr_1
        return gamma_k

    def get_Z(self, la):
        X = self.get_X()
        sum = 0
        for k in range(self.K):
            sum += 1 / self.get_cnr(k, 0)
        frac = self.channel_bandwidth / (2 * la * sum)
        val = X**2 + frac
        Z = X + np.sqrt(val)
        return Z

    def get_X(self):
        numerator = 0
        denominator = 0
        for k in range(self.K):
            cnr_0 = self.get_cnr(k, 0)
            cnr_1 = self.get_cnr(k, 1)
            numerator += (cnr_1 - cnr_0) / (cnr_0 * cnr_1)
            denominator += 1 / cnr_0

        X = numerator / (4 * denominator)
        return X

    def get_sinr(self, channel_idx):
        cnr_0 = self.get_cnr(channel_idx, 0)
        cnr_1 = self.get_cnr(channel_idx, 1)

        A = self.get_A(channel_idx)

        return (A * (A - 1)) / cnr_0 + (A - 1) / cnr_1

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
        distance = self.user_info[user_idx]["distance"]

        # WARN: Deprecated
        # channel_mean = np.mean(self.channels)
        # channel = self.channels[channel_idx]
        # var_channel = np.mean((channel - channel_mean) ** 2)

        rng = np.random.default_rng(seed=self.seed)
        rayleigh_dist = rng.rayleigh(self.channel_variance, distance)

        return rayleigh_dist

    def get_distance_loss(self, user_idx):
        """Distance Loss: d^{-alpha}_n"""
        distance = self.user_info[user_idx]["distance"]
        distance_loss = distance ** (-self.alpha)

        return distance_loss

    def _generate_user(self, idx, seed):
        """
        Generate an user at a random position.

        Returns:
            dict {
                "idx": int,
                "distance": float,
                "power": float,
                "data_rate": float,
            }
        """
        np.random.seed(seed)
        user_dict = {
            "user_idx": idx,
            "distance": np.random.randint(50, 300),
            "power": 0,
            "data_rate": 0,
        }

        return user_dict

    def _update_info(self, steps, indices):
        self.info = {"n_steps": steps, "user_channel_indices": indices}
