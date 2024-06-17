import hydra

from envs.noma_env import NOMA_Env
from model.cnn import CNN
from trainer.trainer import Trainer


@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(cfg):
    device = "cpu"

    env = NOMA_Env(
        num_users=cfg.num_users,
        num_channels=cfg.num_channels,
        metric=cfg.metric,
        device=device,
    )
    env_bl = NOMA_Env(
        num_users=cfg.num_users,
        num_channels=cfg.num_channels,
        device=device,
    )

    model = CNN(
        num_features=3,
        hidden_dim=cfg.hidden_dim,
        num_users=cfg.num_users,
        num_channels=cfg.num_channels,
        device=device,
    )

    trainer = Trainer(
        env=env,
        env_bl=env_bl,
        model=model,
        metric=cfg.metric,
        accelerator=device,
        batch_size=cfg.batch_size,
        num_users=cfg.num_users,
        num_channels=cfg.num_channels,
        num_episodes=cfg.num_episodes,
        num_tests=cfg.num_tests,
        loss_threshold=cfg.loss_threshold,
    )

    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    train()
