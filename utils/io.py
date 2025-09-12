import os
import shutil
import torch

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def save_state_dict(model, path):
    torch.save(model.state_dict(), path)