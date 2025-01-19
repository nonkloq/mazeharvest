import math
import os

import torch
from .ttutil import DEVICE

millnames = ["", " K", "M", " B", " T"]


def millify(n: int):
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )

    return f"{n / 10**(3 * millidx):.2f}{millnames[millidx]}"


def get_param_count(model: torch.nn.Module):
    total_params = 0
    total_trainable_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            total_trainable_params += num_params

    return millify(total_params), millify(total_trainable_params)


def print_param_counts(model: torch.nn.Module):
    """
    Model Parameter count printer
    """
    total_params = 0
    total_trainable_params = 0

    print("Layer-wise parameter counts:")
    print("=" * 50)

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            total_trainable_params += num_params

        print(f"Layer: {name} | Parameters: {num_params}")

    print("=" * 50)
    print(f"Total parameters: {total_params} ({millify(total_params)})")
    print(
        f"Total trainable parameters: {total_trainable_params} ({millify(total_trainable_params)})"
    )
    print("=" * 50)


def _loadcp(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoint file not found: {file_path}")

    return torch.load(file_path, weights_only=False, map_location=DEVICE)


def load_params(model: torch.nn.Module, file_path: str):
    checkpoint = _loadcp(file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    update = checkpoint["update"]
    loss = checkpoint["loss"]

    print(f"Parameters loaded from {file_path} @ iter {update} with loss {loss}")
    return model


def load_checkpoint(trainer, file_path: str):
    checkpoint = _loadcp(file_path)

    trainer.model_ = checkpoint["model_state_dict"]
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    update = checkpoint["update"]
    loss = checkpoint["loss"]

    print(f"Checkpoint loaded from {file_path}")
    print(f"Resuming from Update step {update}, loss: {loss}")

    return trainer


def save_checkpoint(trainer, update, loss, file_name: str):
    save_dir = "/tmp/checkpointdir"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, file_name)
    checkpoint = {
        "update": update,
        # "config": trainer.conf,
        "model_state_dict": trainer.model_.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"New Checkpoint saved at {checkpoint_path}")
