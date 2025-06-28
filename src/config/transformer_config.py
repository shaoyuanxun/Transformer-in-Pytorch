from pathlib import Path


def get_config():
    """
    Returns the configuration for the Transformer model.

    Returns:
        A dictionary containing the model and training parameters.
    """
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 1e-4,
        "max_len": 350,
        "d_model": 512,
        "d_hidden": 2048,
        "n_head": 8,
        "n_layers": 8,
        "drop_prob": 0.1,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "transformer_",
        "preload": "latest",
        "tokenizer_file": "src/tokenizers/tokenizer_{0}.json",
        "experiment_name": "runs/transformer",
    }


def get_weights_file_path(config, epoch: str):
    """
    Constructs the file path for the model weights file for a given epoch.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        epoch (str): The epoch number to construct the weights file path for.

    Returns:
        str: The full file path to the weights file for the specified epoch.
    """

    model_folder = f"src/weights/{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    """
    Finds the latest weights file in the weights folder.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.

    Returns:
        str: The full file path to the latest weights file, or None if there are no weights files.
    """
    model_folder = f"src/weights/{config['datasource']}_{config['model_folder']}"
    model_filename: str = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
