import copy
import torch
import matplotlib.pyplot as plt
import random

from envs.core_env import BaseEnv
from trainer.logger import Logger
from trainer.replay_memory import ReplayMemory
from tqdm import tqdm
from torch.distributions import Categorical


class Trainer:
    def __init__(
        self,
        env: BaseEnv,
        env_bl: BaseEnv,
        model: torch.nn.Module,
        metric: str = "MSR",
        accelerator: str = "cpu",
        batch_size: int = 40,
        num_users: int = 10,
        num_channels: int = 15,
        num_episodes: int = 10000,
        num_tests: int = 10,
        loss_threshold: float = 1e-3,
        epsilon: float = 1,
        save_dir: str = "simulations",
        save_every: int = 10,
        method: str = "Q-Learning",
    ):
        """
        Creates two sets of training-purpose objects: baseline objects and testing objects.
        Each comprises `env` and `model`.
        """
        self.method = method

        self.batch_size = batch_size
        self.N = num_users
        self.K = num_channels

        # Testing objects
        self.env = env
        self.online_model = model  # act randomly based on the given distribution
        self.sync_every = 1e2

        # Baseline objects
        self.target_model = copy.deepcopy(self.online_model)
        self.env_bl = env_bl

        self.metric = metric  # "MSR" or "MMR"
        self.logger = Logger(save_dir=save_dir, save_every=save_every)
        self.buffer = ReplayMemory(batch_size=self.batch_size)

        self.num_episodes = num_episodes
        self.num_tests = num_tests
        self.loss_threshold = loss_threshold
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999975

        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=1e-4)
        self.loss_func = torch.nn.SmoothL1Loss()

        if accelerator not in ["cpu", "mps", "gpu", "cuda"]:
            raise Exception(
                "`accelerator` should be either 'cpu', 'mps', 'cuda', or 'gpu'."
            )
        elif accelerator == "gpu" or accelerator == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise Exception("`cuda` not available")
        elif accelerator == "mps":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                raise Exception("`mps` not available")
        else:  # accelerator == "cpu"
            self.device = torch.device("cpu")

        self.online_model.to(self.device)
        self.target_model.to(self.device)

    def train_test(self):
        sum = []
        episodes = tqdm(range(self.num_episodes))
        for episode in episodes:
            if episode % self.sync_every == 0:
                self.sync_networks()

            state, _ = self.env.reset()
            state = state.unsqueeze(0)

            history = []
            for _ in range(self.N):
                out = self.online_model(state)

                # NOTE: Action (with `Online` network)
                valid_actions_mask = out != float("-inf")
                valid_actions_mask = valid_actions_mask.view(-1)
                valid_indices = torch.nonzero(valid_actions_mask)

                # Exploration
                if torch.rand(1) < self.epsilon:
                    action = random.choice(valid_indices)
                # Exploitation
                else:
                    pred_reward, action = torch.max(out, dim=1)
                self.epsilon = self.epsilon * self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)

                # history.append([prev_state.clone(), state.clone(), torch.tensor([action])])

                # Action
                # (prev_state, state), reward, info, _ = self.env.step(action)
                next_state, reward, info, done = self.env.step(action)
                history.append(
                    [
                        state,
                        next_state,
                        torch.tensor([action]),
                        torch.tensor([done]),
                    ]
                )

                state = next_state.unsqueeze(0)

                # NOTE: Learn
                if self.buffer.get_len() >= 1e2:
                    m_state, m_next_state, m_action, m_done, m_reward = (
                        self.buffer.sample_from_memory()
                    )

                    m_reward = m_reward.squeeze(1).to(self.device)
                    expected_reward = self.td_estimate(m_state, m_action)
                    target_reward = self.td_target(m_reward, m_next_state, m_done)

                    # loss = self.loss_func(expected_reward, m_reward)
                    loss = self.loss_func(expected_reward, target_reward)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            sum_rate = 0
            for i, idx in enumerate(info["usr_idx_history"]):
                usr_info = info["user_info"][idx]
                data_rate = usr_info["data_rate"] / 1e6
                data_rate = torch.tensor([data_rate], dtype=torch.float32)
                sum_rate = sum_rate + data_rate

                # print(idx.item(), usr_info["channel"], data_rate, usr_info["power"], usr_info["distance"], usr_info["CNR"])
                history[i].append(data_rate)
                m_state = history[i][0]
                m_next_state = history[i][1]
                m_action = history[i][2]
                m_done = history[i][3]
                m_reward = history[i][4]
                self.buffer.save_into_memory(
                    m_state, m_next_state, m_action, m_done, m_reward
                )

            if episode % 100 == 0 and self.buffer.get_len() >= 1e2:
                sum.append(sum_rate)
                print(f"EP {episode}: {loss}, {sum_rate}")

        plt.plot(sum)
        plt.show()

    def fit(self):
        episodes = tqdm(range(self.num_episodes))
        avg_sum_rate = 0
        avg_loss = 0
        max_sum_rate = 0
        for episode in episodes:
            state, info = self.env.reset()
            state_bl, info_bl = self.env_bl.reset()
            state = state.unsqueeze(0)
            state_bl = state_bl.unsqueeze(0)

            loss = None
            log_probs = 1
            for _ in range(self.N):
                action, log_prob, action_bl = self.action_select(state, state_bl)
                log_probs = log_probs * log_prob

                next_state, _, info, _ = self.env.step(action)
                next_state_bl, _, info_bl, _ = self.env_bl.step(action_bl)
                state = next_state.unsqueeze(0)
                state_bl = next_state_bl.unsqueeze(0)

            sum_rate = self.calculate_actual_sum_rate(info)
            sum_rate_bl = self.calculate_actual_sum_rate(info_bl)
            loss = self.policy_gradient(log_probs, sum_rate, sum_rate_bl)

            avg_sum_rate += sum_rate
            avg_loss += loss
            max_sum_rate = max(sum_rate, max_sum_rate)

            if sum_rate > sum_rate_bl:
                self.sync_networks()

            if episode % 100 == 0:
                avg_sum_rate /= 100
                avg_loss /= 100
                print(f"EP: {episode}: {avg_loss}, {avg_sum_rate}, {max_sum_rate}")
                self.logger.log_step(value=loss, log="loss")
                self.logger.log_step(value=sum_rate, log="sum_rate")
                avg_sum_rate = 0
                avg_loss = 0

    def action_select(self, state, state_bl):
        probs = self.online_model(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        probs_bl = self.target_model(state_bl)
        dist_bl = Categorical(probs_bl)
        action_bl = dist_bl.sample()

        return action, log_prob, action_bl

    def policy_gradient(self, log_probs, reward, reward_bl):
        loss = (reward - reward_bl) * log_probs

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()

    def action_selection(self, state):
        action_space = self.online_model(state)

        valid_actions_mask = action_space != float("-inf")
        valid_actions_mask = valid_actions_mask.view(-1)
        valid_indices = torch.nonzero(valid_actions_mask)

        # Exploration
        if torch.rand(1) < self.epsilon:
            action = random.choice(valid_indices)
        # Exploitation
        else:
            pred_reward, action = torch.max(action_space, dim=1)

        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        return action

    def learn(self):
        state, next_state, action, done, reward = self.buffer.sample_from_memory()
        td_estimate = self.td_estimate(state, action)
        td_target = self.td_target(reward, next_state, done)

        loss = self.loss_func(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calculate_actual_sum_rate(self, info: dict):
        sum_rate = 0
        for idx in info["usr_idx_history"]:
            usr_info = info["user_info"][idx]
            data_rate = usr_info["data_rate"] / 1e6
            data_rate = torch.tensor([data_rate], dtype=torch.float32)
            sum_rate = sum_rate + data_rate

        return sum_rate

    def train(self):
        T_s = 5  # training stopping criterion
        outperform_count = 0  # when the model outperforms the baseline model, +1
        """Run training process"""

        episodes = tqdm(range(self.num_episodes))
        for episode in episodes:
            state, info = self.env.reset()
            # reward = torch.tensor([])
            self.model._reset()

            # Testing
            # zeta = torch.zeros(state.shape[0], self.K * self.N, 2)
            loss = torch.zeros(state.shape[0])  # create batch-sized loss tensor
            steps = tqdm(
                range(self.N - 1),
                desc="Testing model",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
            for _ in steps:
                action, probability = self.one_step_feedforward(
                    model=self.model, state=state, is_baseline=False
                )
                # zeta.append(state)
                state, reward, info, done = self.env.step(action)

            loss = reward
            loss_log = torch.mean(loss)

            self.logger.log_step(loss_log, "train", "Loss")

            state_bl, info_bl = self.env_bl.reset()
            self.model_bl._reset()

            # Baseline
            loss_bl = torch.zeros(state.shape[0])  # create batch-sized loss tensor
            steps = tqdm(
                range(self.N - 1),
                desc="Baseline model",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
            for _ in steps:
                action_bl, probability_bl = self.one_step_feedforward(
                    model=self.model_bl, state=state_bl, is_baseline=True
                )
                state_bl, reward_bl, info_bl, done_bl = self.env_bl.step(action_bl)

            loss_bl = reward_bl / self.N

            # FIXME: Consider batch
            # if loss > loss_bl:
            #     outperform_count += 1
            # else:
            #     outperform_count = 0
            # if loss < loss_bl:
            #     self.model_bl.load_state_dict(self.model.state_dict())

            if outperform_count >= T_s:
                break
            else:
                # p = self.feedforward_and_get_prob(self.model, state_bl)
                loss = torch.mean(loss - loss_bl * torch.log(probability))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        print(reward)

    def one_step_feedforward(self, model, state, is_baseline: bool = False):
        """
        Model feedforwards one step given some state and transits to a new timestep.
        This is to differ baseline model and testing model.

        Args:
            model (BaseModel(nn.Module))
            state
            is_baseline (bool=False): to differ baseline model and testing model.

        Returns:
            chosen_state
            chosen_user_idx
            chosen_channel_idx
        """
        policy = model(state)

        if is_baseline:
            action = torch.argmax(policy, dim=1)
        else:
            action = torch.multinomial(policy, 1).squeeze(1)

        probability = policy[torch.arange(policy.size(0)), action]

        return action, probability

    @torch.no_grad()
    def feedforward_and_get_prob(self, model, state):
        policy = model(state)
        prob = torch.max(policy)

        return prob

    def td_estimate(self, state, action):
        action = action.squeeze()
        return self.online_model(state)[torch.arange(0, self.batch_size), action]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        reward = reward.to(self.device).squeeze(1)
        done = done.to(self.device).squeeze(1)
        next_state_reward = self.target_model(next_state)
        pred_reward, action = torch.max(next_state_reward, dim=1)
        # next_reward = self.online_model(state, next_state)

        next_reward = self.online_model(next_state)[
            torch.arange(0, self.batch_size), action
        ]
        next_reward = torch.where(done, torch.tensor(0), next_reward)
        val = reward + next_reward
        return val

    def sync_networks(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def test(self):
        """Run testing process"""
        for episode in range(self.num_tests):
            (prev_state, state), _ = self.env.reset()
            prev_state = prev_state.unsqueeze(0)
            state = state.unsqueeze(0)
            loss_reward = torch.zeros(state.shape[0], 1)
            self.online_model._reset()

            steps = tqdm(
                range(self.N - 1),
                desc="Testing model",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
            history = []
            for _ in steps:
                out = self.online_model(prev_state, state)

                pred_reward, action = torch.max(out, dim=1)

                # Real Action
                (prev_state, state), reward, info, _ = self.env.step(action)
                prev_state = prev_state.unsqueeze(0)
                state = state.unsqueeze(0)

            sum_rate = 0
            for i, info in enumerate(info["user_info"]):
                data_rate = info["data_rate"] / 1e6
                sum_rate = sum_rate + data_rate

            print(f"EP {episode}: {sum_rate}")
        exit()
