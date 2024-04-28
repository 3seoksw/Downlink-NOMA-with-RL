import hydra

from model import DQN


@hydra.main(version_base=None, config_path="config", config_name="train")
def train(cfg):
    """Training code goes here"""
    model = DQN(input_dim=cfg.input_dim, output_dim=cfg.output_dim)


if __name__ == "__main__":
    train()
