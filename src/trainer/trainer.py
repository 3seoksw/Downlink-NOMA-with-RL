import torch

from envs.wireless_communication import WirelessCommunication
from model.base_model import BaseModel


class Trainer:
    def __init__(
        self,
        env: WirelessCommunication,
        model: BaseModel,
        accelerator: str = "cpu",
        num_episodes: int = 10000,
        num_tests: int = 10,
    ):
        self.env = env
        self.model = model
        self.num_episodes = num_episodes
        self.num_tests = num_tests

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
        """Run training process"""
        for episode in range(self.num_episodes):
            state, info = self.env.reset()

            action = self.model(state)
            obs, reward, info, done = self.env.step(action)

    def test(self):
        """Run testing process"""
