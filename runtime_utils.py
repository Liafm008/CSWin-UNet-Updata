import os
import random
from collections import OrderedDict

import numpy as np
import torch


def resolve_device(device_name=None):
    if device_name:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_runtime(seed, deterministic=True, use_tf32=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")

    import torch.backends.cudnn as cudnn

    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic


def resolve_data_dir(path, expected_leaf):
    normalized = os.path.normpath(path)
    if os.path.basename(normalized) == expected_leaf:
        return path
    candidate = os.path.join(path, expected_leaf)
    if os.path.exists(candidate) or not os.path.exists(path):
        return candidate
    return path


def get_autocast(device, enabled):
    return torch.autocast(device_type=device.type, enabled=enabled)


def create_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict_ema", "state_dict", "model"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                checkpoint = state_dict
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    clean_state_dict = OrderedDict()
    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith("module.") else key
        clean_state_dict[clean_key] = value
    return clean_state_dict


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    return checkpoint, state_dict


def save_checkpoint(path, model, optimizer=None, scaler=None, epoch=None, iter_num=None, best_performance=None):
    checkpoint = {
        "state_dict": unwrap_model(model).state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if iter_num is not None:
        checkpoint["iter_num"] = iter_num
    if best_performance is not None:
        checkpoint["best_performance"] = best_performance

    torch.save(checkpoint, path)
