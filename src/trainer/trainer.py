import copy
import torch
import numpy as np

from envs.core_env import BaseEnv
from model.base_model import BaseModel
from trainer.logger import Logger
from model.ann import ANN
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        env: BaseEnv,
        env_bl: BaseEnv,
        model: BaseModel,
        metric: str = "MSR",
        accelerator: str = "cpu",
        num_users: int = 10,
        num_channels: int = 15,
        num_episodes: int = 10000,
        num_tests: int = 10,
        loss_threshold: float = 1e-3,
        save_dir: str = "simulations",
        save_every: int = 10,
        method: str = "Q-Learning",
    ):
        """
        Creates two sets of training-purpose objects: baseline objects and testing objects.
        Each comprises `env` and `model`.
        """
        self.method = method

        self.N = num_users
        self.K = num_channels

        # Testing objects
        self.env = env
        self.model = model  # act randomly based on the given distribution

        # Baseline objects
        self.env_bl = env_bl
        self.model_bl = copy.deepcopy(self.model)  # act greedily

        self.metric = metric  # "MSR" or "MMR"
        self.logger = Logger(save_dir=save_dir, save_every=save_every)
        self.num_episodes = num_episodes
        self.num_tests = num_tests
        self.loss_threshold = loss_threshold

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_func = torch.nn.MSELoss()

        if accelerator not in ["cpu", "mps", "gpu"]:
            raise Exception("`accelerator` should be either 'cpu', 'mps', or 'gpu'.")
        elif accelerator == "gpu":
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

        self.model.to(self.device)

        self.online_model = ANN()
        self.target_model = ANN().load_state_dict(self.online_model.state_dict())

    def fit(self):
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            loss_reward = torch.zeros(state.shape[0], 1)
            self.model._reset()

            pred_rewards = []
            steps = tqdm(
                range(self.N - 1),
                desc="Testing model",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
            for _ in steps:
                out = self.model(state)
                pred_reward, action = torch.max(out, dim=1)
                pred_rewards.append(pred_reward)

                next_state, reward, _, _ = self.env.step(action)
                state = next_state

            loss_reward = reward.float()
            loss_reward.to(self.device)

            pred_rewards = torch.stack(pred_rewards)
            if self.metric == "MSR":
                pred_loss_reward = torch.sum(pred_rewards, dim=0)
            elif self.metric == "MMR":
                pred_loss_reward = torch.min(pred_rewards, dim=0)
            else:
                raise KeyError()

            loss = self.loss_func(pred_loss_reward, loss_reward)
            print(episode, loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

    def test(self):
        """Run testing process"""
