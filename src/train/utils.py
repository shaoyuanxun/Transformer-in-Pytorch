import torch


def get_device():
    """
    Get the device to use for training.
    Returns:
        torch.device: The device to use for training.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.has_mps or torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
