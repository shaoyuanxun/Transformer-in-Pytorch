from src.config.transformer_config import get_config
from src.train.transformer_train import train_model

if __name__ == "__main__":
    config = get_config()
    train_model(config)
    