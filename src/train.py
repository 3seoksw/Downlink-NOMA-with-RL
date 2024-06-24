import hydra

from hydra.core.hydra_config import HydraConfig
from envs.noma_env import NOMA_Env
from model.cnn import CNN
from model.linear import FCNN
from model.ann import ANN
from trainer.trainer import Trainer


@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(cfg):
    save_dir = HydraConfig.get().runtime.output_dir

    env = NOMA_Env(
        input_dim=cfg.input_dim,
        num_users=cfg.num_users,
        num_channels=cfg.num_channels,
        metric=cfg.metric,
        device=cfg.device,
    )
    env_bl = NOMA_Env(
        input_dim=cfg.input_dim,
        num_users=cfg.num_users,
        num_channels=cfg.num_channels,
        metric=cfg.metric,
        device=cfg.device,
    )

    if cfg.model == "CNN":
        model = CNN(
            num_features=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_users=cfg.num_users,
            num_channels=cfg.num_channels,
            device=cfg.device,
        )
    elif cfg.model == "FCNN":
        model = FCNN(
            num_features=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_users=cfg.num_users,
            num_channels=cfg.num_channels,
            device=cfg.device,
        )
    elif cfg.model == "ANN":
        model = ANN(
            num_features=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_users=cfg.num_users,
            num_channels=cfg.num_channels,
            device=cfg.device,
        )
    else:
        raise KeyError("Choose the model either `CNN`, `FCNN`, or `ANN`.")

    trainer = Trainer(
        env=env,
        env_bl=env_bl,
        model=model,
        metric=cfg.metric,
        accelerator=cfg.device,
        batch_size=cfg.batch_size,
        num_users=cfg.num_users,
        num_channels=cfg.num_channels,
        num_episodes=cfg.num_episodes,
        num_tests=cfg.num_tests,
        T=cfg.T,
        error_threshold=cfg.error_threshold,
        loss_threshold=cfg.loss_threshold,
        save_dir=save_dir,
        learning_rate=cfg.learning_rate,
        validation_seeds=cfg.validation_seeds,
        validate_every=cfg.validate_every,
    )

    trainer.fit()
    # trainer.test()


if __name__ == "__main__":
    train()
