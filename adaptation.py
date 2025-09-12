import argparse
import random
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import datetime
import wandb

from train.trainer import Trainer
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    seed: int = field(
        default=1234,
        metadata={
            "help":"Random seed used for trainig"
        }
    )
    batch_size: int = field(
        default=2,
        metadata={
            "help":"Batchsize for training"
        }
    )
    num_epochs: int = field(
        default=1,
        metadata={
            "help":"Number of training epochs"
        }
    )
    lr: float = field(
        default=2e-6,
        metadata={
            "help":"Learning rate"
        }
    )
    alpha: float = field(
        default=0.1,
        metadata={
            "help":"Weight for prototype pull loss"
        }
    )
    backbone: str = field(
        default="sam2_s",
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
    patch_stride_src: int = field(
        default=128,
        metadata={
            "help":"Stride for sliding window operation in source domain"
        }
    )
    patch_stride_tgt: int = field(
        default=384,
        metadata={
            "help":"Stride for sliding window operation in target domain"
        }
    )
    warmup_pl: int = field(
        default=2000,
        metadata={
            "help":"Number of pseudolabel generated for warmup epoch"
        }
    )
    iters_pl: int = field(
        default=200,
        metadata={
            "help":"Number of pseudolabel generated for each epoch"
        }
    )
    uc_threshold: float = field(
        default=0.85,
        metadata={
            "help":"Threshold for uncertainty mask generation"
        }
    )
    data_root: str = field(
        default="data",
        metadata={
            "help":"Root of training data"
        }
    )
    pd_root: str = field(
        default="data_pd_cg_src_2000_01_ori",
        metadata={
            "help":"Root of pseudolabel data"
        }
    )
    tgt_ckpt: str = field(
        default="exp2/Cityscapes/best_sam2_s_10.pth",
        metadata={
            "help": "target model checkpoint path"
        }
    )
    src_ckpt: str = field(
        default="exp2/Cityscapes/best_sam2_s_10.pth",
        metadata={
            "help": "source model checkpoint path"
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
    set_seed(args.seed)
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(
        project="OmniSAM",
        name=dt,
        group=args.backbone + "_" + args.dataset,
        config=vars(args)
    )
    engine = Engine(args)
    engine.setup()

    for epoch in range(args.num_epochs):
        engine.generate_pseudo_labels(epoch)
        engine.train(epoch)
        engine.validate(epoch)
    
    wandb.finish()
    print('Training Finished.')

if __name__ == "__main__":
    main()