import argparse
import random
import warnings
from dataclasses import dataclass, field, asdict

import os
import numpy as np
import torch
import torch.distributed as dist
import datetime
import wandb

from train.trainer import Trainer
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class TrainConfig:
    seed: int = field(
        default=1234,
        metadata={
            "help":"Random seed used for trainig"
        }
    )
    batch_size: int = field(
        default=8,
        metadata={
            "help":"Batchsize for training"
        }
    )
    num_epochs: int = field(
        default=10,
        metadata={
            "help":"Number of training epochs"
        }
    )
    lr: float = field(
        default=1e-4,
        metadata={
            "help":"Learning rate"
        }
    )
    backbone: str = field(
        default="sam2_t",
        metadata={
            "help":"Backbone to use for training",
            "choices": ["sam2_t", "sam2_s", "sam2_b+", "sam2_l"],
        }
    )
    dataset: str = field(
        default="Cityscapes",
        metadata={
            "help":"Dataset to use for training",
            "choices": ["Cityscapes", "Stanford2D3D", "SynPASS"],
        }
    )
    num_maskmem: int = field(
        default=3,
        metadata={
            "help":"Capacity of memory bank"
        }
    )
    image_size: int = field(
        default=1024,
        metadata={
            "help":"Width (height) of the image patches"
        }
    )
    patch_stride: int = field(
        default=128,
        metadata={
            "help":"Stride for sliding window operation during training"
        }
    )
    data_root: str = field(
        default="data",
        metadata={
            "help":"Root of training data"
        }
    )
    load_ckpt: str = field(
        default=None,
        metadata={
            "help":"Load checkpoints before training"
        }
    )
    save_ckpt_dir: str = field(
        default="checkpoints",
        metadata={
            "help":"Directory for saving checkpoints"
        }
    )
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_arg() -> TrainConfig:
    parser = argparse.ArgumentParser(description="OmniSAM Training (Refactored)")
    for f in TrainConfig.__dataclass_fields__.values():
        if f.type == bool:
            parser.add_argument(f"--{f.name}", action="store_true" if f.default is False else "store_false")
        else:
            parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = parser.parse_args()
    return TrainConfig(**vars(args))

def main():
    args = parse_arg()
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(args.seed)
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if rank == 0:
        wandb.init(
            project="OmniSAM_src_train",
            name=dt,
            group=args.backbone + "_" + args.dataset,
            config=vars(args)
        )
    trainer = Trainer(args, device, local_rank, world_size)
    trainer.setup()

    for epoch in range(args.num_epochs):
        trainer.train(epoch)
        trainer.validate(epoch)
    
    wandb.finish()
    dist.destroy_process_group()
    print('Training Finished.')

if __name__ == "__main__":
    main()