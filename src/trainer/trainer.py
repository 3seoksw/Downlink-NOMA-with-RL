import copy
import torch
import numpy as np

from envs.noma_env import NOMA_Env
from model.base_model import BaseModel
from trainer.logger import Logger


class Trainer:
    def __init__(
        self,
        env: NOMA_Env,
        model: BaseModel,
        logger: Logger,
        metric: str = "MSR",
        accelerator: str = "cpu",
        num_users: int = 10,
        num_channels: int = 15,
        num_episodes: int = 10000,
        num_tests: int = 10,
        loss_threshold: float = 1e-3,
    ):
        """
        Creates two sets of training-purpose objects: baseline objects and testing objects.
        Each comprises `env` and `model`.
        """
        self.N = num_users
        self.K = num_channels

        # Testing objects
        self.env = env
        self.model = model  # act randomly based on the given distribution

        # Baseline objects
        self.env_bl = copy.deepcopy(env)
        self.model_bl = copy.deepcopy(self.model)  # act greedily

        self.metric = metric  # "MSR" or "MMR"
        self.logger = logger
        self.num_episodes = num_episodes
        self.num_tests = num_tests
        self.loss_threshold = loss_threshold

        self.optimizer = torch.optim.Adam(self.model.parameters())

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

    def train(self):
        T_s = 5  # training stopping criterion
        outperform_count = 0  # when the model outperforms the baseline model, +1
        """Run training process"""
        for episode in range(self.num_episodes):
            state, info = self.env.reset()
            state_bl = state

            zeta = []
            G_MSR = 0  # Maximizing Sum Rate Goal
            G_MMR = -1  # Maximizing Minimal Rate Goal
            zeta_bl = []
            G_MSR_bl = 0
            G_MMR_bl = 100
            loss = torch.zeros(state.shape[0])
            for n in range(self.env.get_rx()):
                # Testing
                state, user_idx, channel_idx = self.one_step_feedforward(
                    model=self.model, state=state, is_baseline=False
                )
                zeta.append(state)
                action = (user_idx, channel_idx)
                state, reward, done, info = self.env.step(action)

                if self.metric == "MSR":
                    loss += reward
                elif self.metric == "MMR":
                    loss = torch.min(loss, reward)
                else:
                    raise KeyError("`MSR` or `MMR` available.")

                # Baseline
                state_bl, user_idx_bl, channel_idx_bl = self.one_step_feedforward(
                    model=self.model_bl, state=state_bl, is_baseline=False
                )
                zeta_bl.append(state_bl)
                action_bl = (user_idx_bl, channel_idx_bl)
                state_bl, reward_bl, done_bl, info_bl = self.env_bl.step(action_bl)
                G_MSR_bl += reward_bl
                G_MMR_bl = min(G_MMR_bl, reward)

                if self.metric == "MSR":
                    loss_bl += reward_bl
                elif self.metric == "MMR":
                    loss_bl = torch.min(loss_bl, reward_bl)
                else:
                    raise KeyError("`MSR` or `MMR` available.")

            if loss > loss_bl:
                outperform_count += 1
            else:
                outperform_count = 0

            if outperform_count >= T_s:
                break
            else:
                # WARN: zeta might require concatenation
                p = self.feedforward_and_get_prob(self.model, zeta)
                loss = torch.mean(loss - loss_bl * torch.log(p))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # FIXME: Erase the following after the above algorithm
            # Loss and logging
            loss = torch.tensor(-loss)
            loss_bl = torch.tensor(-loss_bl)
            if outperform_count >= T_s:
                break
            else:  # TODO: Adam()
                grad_policy
                grad_loss = loss - loss_bl * 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if loss < loss_bl:
                    self.model_bl.load_state_dict(self.model.state_dict())

    def one_step_feedforward(self, model, state, is_baseline: bool = True):
        """
        Model feedforwards one step given some state and transits to a new timestep.
        This is to differ baseline model and testing model.

        Args:
            model (BaseModel(nn.Module))
            state
            is_baseline (bool=True): to differ baseline model and testing model.

        Returns:
            chosen_state
            chosen_user_idx
            chosen_channel_idx
        """
        policy = model(state)

        if is_baseline:
            chosen_state = np.argmax(policy)
        else:
            chosen_state = np.random.choice(len(policy), p=policy)

        channel_idx = chosen_state / self.N
        user_idx = chosen_state - self.N * channel_idx

        return chosen_state, user_idx, channel_idx

    @torch.no_grad()
    def feedforward_and_get_prob(self, model, state):
        policy = model(state)
        prob = np.max(policy)

        return prob

    def test(self):
        """Run testing process"""
