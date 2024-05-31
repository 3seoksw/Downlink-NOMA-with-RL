import torch
import numpy as np

from collections import deque
from typing import Optional
from envs.core_env import BaseEnv


class NOMA_Env(BaseEnv):
    """
    ### State Space
    The state is a pair of user and channel.
    For instance, s_t = (U_n, C_k) represents n-th user has been assigned to k-th channel.
    N and K represents the number of users and channels, respectively.
    Hence, the state space contains NK states and the size of it is (NK, 2).
    States are in the form of the following:

           |      U_0       |      U_1       |      U_2       | ... |      U_{N-1}
    _______|________________|________________|________________|_____|__________________
      C_0  |   (U_0, C_0)   |   (U_1, C_0)   |   (U_2, C_0)   | ... |  (U_{N-1}, C_0})
      C_1  |   (U_0, C_1)   |   (U_1, C_1)   |   (U_2, C_1)   | ... |  (U_{N-1}, C_1})
       .   |                |                |                |     |
       .   |                |                |                |     |
       .   |                |                |                |     |
    C_{K-1}| (U_0, C_{K-1}) | (U_1, C_{K-1}) | (U_2, C_{K-1}) | ... | (U_{N-1}, C_{K-1})

    Each state's value is consist of user and channel information, distance and CNR, repectively.

    ### Action Space
    The action from this NOMA environment is to assign a channel to a user.
    Here, we suppose the maximum number of users which can be assigned to a channel is 2.
    The action is in the form of `(user_index, channel_index)`.
    """

    def __init__(
        self,
        input_dim: int = 3,
        num_users: int = 10,
        num_channels: int = 5,
        B_tot: float = 5e6,
        alpha: int = 2,
        P_T: int = 12,
        seed: Optional[float] = None,
        N_0: int = -170,
        min_data_rate: int = 2,
        metric: str = "MSR",
        device: str = "cpu",
    ):
        super().__init__(device)

        self.device_name = device
        self.input_dim = input_dim
        self.N = num_users  # 40
        self.K = num_channels  # 20
        self.bandwidth_total = B_tot  # 5 MHz (5e6 Hz)
        self.channel_bandwidth = self.bandwidth_total / self.K  # 2,500,000 Hz
        self.alpha = alpha  # path loss coefficient (alpha=2)
        self.total_power = P_T  # 2 ~ 12 Watt
        self.seed = seed  # 2024
        self.noise = N_0  # -170 dBm/Hz
        self.channel_variance = (
            self.bandwidth_total * 10 ** (self.noise / 10) * 1e-3 / self.K
        )  # sigma_{z_k}^2
        self.min_data_rate = min_data_rate  # 2 bps/Hz
        self.metric = metric

        self.states = torch.zeros(self.K * self.N, self.input_dim).to(self.device)
        self.info = {"n_steps": 0, "usr_idx_history": [], "user_info": []}
        self.done = False
        self.prev_step = 0
        self.prev_user = 0

        # key: channel_idx, value: list[(user_idx0, cnr0), (user_idx1, cnr1)]
        self.channel_info = {}

        # NOTE: See `_generate_user()` for more information
        self.user_info = []

    def reset(self, seed: Optional[int] = None):
        """
        Reset the NOMA environment.

        Returns:
            state: an initial state with the size of NK filled with 0s.
        """
        for i in range(self.N):
            user_dict = self._generate_user(i, seed)
            self.user_info.append(user_dict)

        self.channel_info = {}
        self.info = {"n_steps": 0, "usr_idx_history": [], "user_info": []}

        self.done = False

        self.states = torch.zeros(self.K * self.N, self.input_dim).to(self.device)

        for nk in range(self.N * self.K):
            # channel_idx = nk // self.N
            user_idx = nk % self.N
            self.states[nk, 0] = self.user_info[user_idx]["distance"]
            cnr = self.get_cnr_by_usr(user_idx)
            self.user_info[user_idx]["CNR"] = cnr
            self.states[nk, 1] = cnr

        # self.prev_state = self.states.clone()
        # prev = torch.zeros(self.K * self.N, self.input_dim).to(self.device)

        # user_idx = np.random.randint(self.N)
        # channel_idx = np.random.randint(self.K)
        # random_action = user_idx + self.N * channel_idx
        # self.prev_step = random_action
        # self.prev_user = user_idx
        # self.step(random_action)

        # states = (prev, self.states)
        return self.states.clone(), self.info

    def step(self, action):
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
        # self.states[self.prev_step] = self.user_info[self.prev_user]["data_rate"]
        self.info["n_steps"] += 1

        channel_idx = action // self.N
        user_idx = action % self.N
        self.user_info[user_idx]["channel"] = channel_idx
        self.allocate_resources(channel_idx, user_idx)

        # if self.info["n_steps"] != 1:
        self.info["usr_idx_history"].append(user_idx)

        # States update
        # nk = channel_idx * self.N + user_idx
        nk = action
        # self.states[nk, 0] = self.user_info[user_idx]["distance"]
        # self.states[nk, 1] = self.user_info[user_idx]["CNR"]
        self.states[nk, 2] = 1

        curr_state = self.states.clone()

        reward = self.user_info[user_idx]["data_rate"]

        if self.info["n_steps"] == self.N:
            self.done = True
            self.allocate_power()
            if self.metric == "MSR":
                sum_rate = 0
                for i in range(self.N):
                    data_rate = self.user_info[i]["data_rate"] / 1e6
                    sum_rate = sum_rate + data_rate
                reward = sum_rate
                self.info["user_info"] = self.user_info
            elif self.metric == "MMR":
                min_data_rate = self.user_info[0]["data_rate"]
                for i in range(self.N):
                    data_rate = self.user_info[i]["data_rate"]
                    min_data_rate = min(min_data_rate, data_rate)
                reward = min_data_rate / 1e6

        # states = (self.prev_state, self.states)
        # self.prev_state = self.states.clone()

        return (curr_state, reward, self.info, self.done)

    def clone(self):
        return NOMA_Env(self.device_name, env_kwargs=self.env_kwargs)

    def allocate_power(self):
        """Allocate powers to all users and retrieve data rate"""
        for k in range(self.K):
            p_0, p_1 = self.get_power(k)
            power = [p_0, p_1]
            users = self.channel_info[k]
            for i, (n, p) in enumerate(zip(users, power)):
                self.set_power(n, p)
                data_rate = self.get_data_rate(k, i)
                self.set_data_rate(n, data_rate)

    def allocate_resources(self, channel_idx, user_idx):
        channel_idx = int(channel_idx)
        user_idx = int(user_idx)
        if self.channel_info.get(channel_idx) is None:
            self.channel_info[channel_idx] = []
            self.channel_info[channel_idx].append(user_idx)
            cnr = self.get_cnr(channel_idx, 0)
            # power, p1 = self.get_power(channel_idx, self.metric)
            # data_rate = self.get_data_rate(channel_idx, 0, self.metric)
        else:
            self.channel_info[channel_idx].append(user_idx)
            cnr = self.get_cnr(channel_idx, 1)
            # p0, power = self.get_power(channel_idx, self.metric)
            # data_rate = self.get_data_rate(channel_idx, 1, self.metric)

        self.set_cnr(user_idx, cnr)
        # self.set_power(user_idx, power)
        # self.set_data_rate(user_idx, data_rate)

    def set_cnr(self, user_idx, cnr):
        self.user_info[user_idx]["CNR"] = cnr

    def set_data_rate(self, user_idx, data_rate):
        self.user_info[user_idx]["data_rate"] = data_rate

    def get_data_rate(self, channel_idx, n):
        power_0, power_1 = self.get_power(channel_idx)
        cnr_0 = self.get_cnr(channel_idx, 0)
        channel = self.channel_bandwidth

        if n == 0:
            return channel * np.log2(1 + power_0 * cnr_0)
        else:  # n == 1
            cnr_1 = self.get_cnr(channel_idx, 1)
            return channel * np.log2(1 + (power_1 * cnr_1) / (1 + power_0 * cnr_1))

    def set_power(self, user_idx, power):
        self.user_info[user_idx]["power"] = power

    def get_power(self, channel_idx):
        cnr0 = self.get_cnr(channel_idx, 0)
        cnr1 = self.get_cnr(channel_idx, 1)
        la = self.find_lambda(power=self.total_power, metric=self.metric)

        if self.metric == "MSR":
            A = self.get_A(channel_idx)
            q_k = self.get_msr_power_budget(cnr0, cnr1, A, la)
            p_0 = (cnr1 * q_k - A + 1) / (A * cnr1)
            p_1 = q_k - p_0
            return (p_0, p_1)
        elif self.metric == "MMR":
            q_k = self.get_mmr_power_budget(cnr0, cnr1, la)
            p_0 = -(cnr0 + cnr1) + np.sqrt(
                (cnr0 + cnr1) ** 2 + 4 * cnr0 * (cnr1) ** 2 * q_k
            )
            p_1 = q_k - p_0
            return (p_0, p_1)
        else:
            raise KeyError("No such metric is available. Choose either `MMR` or `MSR`.")

    def get_msr_power_budget(self, cnr0, cnr1, A, la):
        """Calculate q^k under MSR metric"""
        B_c = self.bandwidth_total / self.K
        q_k = (B_c / la) - (A / cnr0) + (A / cnr1) - (1 / cnr1)

        return q_k

    def get_mmr_power_budget(self, cnr0, cnr1, la):
        """Calculate q^k under MMR metric"""
        z = self.get_Z(la)
        q_k = (z * cnr1 + cnr0) * (z - 1) / (cnr0 * cnr1)

        return q_k

    def find_lambda(
        self,
        power=12,
        start=-1e10,
        end=1e10,
        epsilon=1e-10,
        threshold=1e-5,
        metric: str = "MSR",
    ):
        """Execute bisection method to find lagrangian coefficient"""
        while True:
            la = (start + end + epsilon) / 2
            sum_q_k = 0
            for channel_idx in range(self.K):
                A = self.get_A(channel_idx)
                cnr0 = self.get_cnr(channel_idx, 0)
                cnr1 = self.get_cnr(channel_idx, 1)
                if metric == "MSR":
                    q_k = self.get_msr_power_budget(cnr0, cnr1, A, la)
                elif metric == "MMR":
                    q_k = self.get_mmr_power_budget(cnr0, cnr1, la)
                else:
                    raise KeyError(
                        "No such metric is available. Choose either `MMR` or `MSR`."
                    )
                sum_q_k += q_k

            if np.abs(sum_q_k - power) < threshold or sum_q_k == power:
                return la

            if sum_q_k > power:
                start = la
            else:
                end = la

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

    def get_A(self, channel_idx):
        # channel_bandwidth = channel_idx * self.channel_bandwidth
        return 2**2

    def get_cnr_by_usr(self, user_idx):
        h = self.get_channel_response_by_usr(user_idx)
        cnr = np.abs(h) ** 2 / self.channel_variance
        return cnr

    def get_cnr(self, channel_idx, n: int = 0):
        """
        CNR (channel-to-noise-ratio):
            Gamma^k_n = |h^k_n|^2 / sigma^2_{z_k}
        """
        # user_idx = self.channel_info[channel_idx][n]
        h = self.get_channel_response(channel_idx, n)
        cnr = np.abs(h) ** 2 / self.channel_variance

        return cnr

    def get_channel_response_by_usr(self, user_idx):
        g = self.sample_from_rayleigh_distribution()
        d = self.get_distance_loss(user_idx)
        h = g * d
        return h

    def get_channel_response(self, channel_idx, n: int = 0):
        """
        h^k_n = g^k_n * d^{-alpha}_n,
        where `g` follows the Rayleigh distribution,
        and `d` is the distance between n-th user and the base station.

        n is either 0 or 1.
        """
        user_idx = self.channel_info[channel_idx][n]
        g = self.sample_from_rayleigh_distribution()
        d = self.get_distance_loss(user_idx)
        h = g * d

        return h

    def sample_from_rayleigh_distribution(self):
        """
        Rayleight Fading: g^{k}_{n},
        f(x; sigma) = x / sigma^2 * e^{-x^2 / 2 sigma^2}, x >= 0

        Args:
            channel_idx: k
            `n` is either 0 or 1 denoting whether the user has been assigned first
            or not.
        """

        rng = np.random.default_rng(seed=self.seed)
        rayleigh_dist = rng.rayleigh()

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
        # np.random.seed(seed)
        user_dict = {
            "user_idx": idx,
            "distance": np.random.randint(50, 300),
            "power": 0,
            "data_rate": 0,
            "CNR": 0,
            "history_idx": -1,
            "channel": -1,
        }

        return user_dict

    def _update_info(self, steps):
        self.info = {"n_steps": steps}
