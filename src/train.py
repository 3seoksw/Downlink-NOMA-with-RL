import hydra
import torch

from envs.noma_env import NOMA_Env
from envs.vector_env import VectorizedEnv
from model.ann import ANN
from trainer.trainer import Trainer


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(cfg):
    device = "cpu"

    env = NOMA_Env(device)
    env = VectorizedEnv(env=env, num_envs=cfg.batch_size)

    model = ANN(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        batch_size=cfg.batch_size,
        num_users=cfg.num_users,
        num_channels=cfg.num_channels,
    )

    trainer = Trainer(
        env=env,
        model=model,
        metric=cfg.metric,
        accelerator="cpu",
        num_users=cfg.num_users,
        num_channels=cfg.num_channels,
        num_episodes=cfg.num_episodes,
        num_tests=cfg.num_tests,
        loss_threshold=cfg.loss_threshold,
    )

    trainer.train()


if __name__ == "__main__":
    train()
