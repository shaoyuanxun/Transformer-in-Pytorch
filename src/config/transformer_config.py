from pathlib import Path


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
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
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
