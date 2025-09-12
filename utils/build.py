import os
import random
import time
from typing import Tuple, List, Dict

import torch
from torch.utils.data import DataLoader

from sam.sam.build_sam import build_sam2
from model.model import OmniSAM, OmniSAM_adapt

from dataset.Cityscapes import Cityscapes13
from dataset.WildPASS import WildPASSforPL, WildPASSforAdaptation
from dataset.DensePASS import DensePASS13 
from dataset.SynPASS import SynPASS13
from dataset.StanfordPin import StanfordPin8DataSet
from dataset.StanfordPan import StanfordPan8forPL, StanfordPan8forAdaptation, StanfordPan8forVal


DATASET_NAME2CLASSES = {
    'Stanford2D3D': ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"],
    'Cityscapes': ["road", "sidewalk", "building", "wall", "fence", "pole",
                   "traffic light", "traffic sign", "vegetation", "terrain",
                   "sky", "person", "car"],
    'SynPASS': ["road", "sidewalk", "building", "wall", "fence", "pole",
                "traffic light", "traffic sign", "vegetation", "terrain",
                "sky", "person", "car"],
    'WildPASS': ["road", "sidewalk", "building", "wall", "fence", "pole",
                 "traffic light", "traffic sign", "vegetation", "terrain",
                 "sky", "person", "car"],
}

# backbone registry: backbone -> (checkpoint_path, yaml_cfg_path)
SAM_CFGS: Dict[str, Tuple[str, str]] = {
    "sam2_l": (
        "sam/checkpoints/sam2.1_hiera_large.pt",
        "/configs/sam2.1/sam2.1_hiera_l.yaml",
    ),
    "sam2_b+": (
        "sam/checkpoints/sam2.1_hiera_base_plus.pt",
        "/configs/sam2.1/sam2.1_hiera_b+.yaml",
    ),
    "sam2_s": (
        "sam/checkpoints/sam2.1_hiera_small.pt",
        "/configs/sam2.1/sam2.1_hiera_s.yaml",
    ),
    "sam2_t": (
        "sam/checkpoints/sam2.1_hiera_tiny.pt",
        "/configs/sam2.1/sam2.1_hiera_t.yaml",
    ),
}

def build_model(args, num_classes, num_maskmem, device):
    sam2_checkpoint, model_cfg = SAM_CFGS[args.backbone]
    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
    model = OmniSAM_adapt(sam, num_classes=num_classes, num_maskmem=num_maskmem).to(device)
    return model

def load_checkpoint(model, ckpt_path):
    if ckpt_path is None or ckpt_path == '':
        return
    state = torch.load(ckpt_path, map_location='cpu')
    new_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)

class DatasetBuilder:
    def __init__(self, args):
        self.args = args

    # ------------------------------
    def build_tgt_for_pl(self):
        if self.args.dataset in ["SynPASS", "Cityscapes"]:
            dataset_name = 'WildPASS'
            class_names = DATASET_NAME2CLASSES[dataset_name]
            num_classes = len(class_names)
            root_syn = os.path.join(self.args.data_root, 'WildPASS')
            val_list = 'dataset/densepass_list/train.txt'
            tgt_train_dataset = WildPASSforPL(root_syn, val_list, crop_size=(4096, 1024), sliding_window=(384, 1024))
            tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=1, shuffle=True,
                                          num_workers=4, pin_memory=True)
        elif self.args.dataset == 'Stanford2D3D':
            class_names = DATASET_NAME2CLASSES['Stanford2D3D']
            num_classes = len(class_names)
            root_syn = os.path.join(self.args.data_root, 'Stanford2d3d_Seg')
            val_list = 'dataset/s2d3d_pan_list/train.txt'
            tgt_train_dataset = StanfordPan8forPL(root_syn, val_list, crop_size=(3072, 1024), sliding_window=(256, 1024), set='train')
            tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=1, shuffle=True,
                                          num_workers=4, pin_memory=True)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        frame_cnt = tgt_train_dataset._get_frame_cnt()
        return tgt_train_loader, class_names, num_classes, frame_cnt

    def build_tgt_from_pd(self, pd_dir: str):
        if self.args.dataset == 'Stanford2D3D':
            root_val = os.path.join(self.args.data_root, 'Stanford2d3d_Seg')
            val_list = 'dataset/s2d3d_pin_list/val.txt'
            target_dataset = StanfordPan8forAdaptation(pd_dir,
                                                os.path.join(os.path.dirname(pd_dir), 'used_path.txt'),
                                                crop_size=(3072, 1024),
                                                sliding_window=(256, 1024))
            val_dataset = StanfordPan8forVal(root_val, val_list, crop_size=(3072, 1024), sliding_window=(256, 1024), set='val')

        elif self.args.dataset in ["SynPASS", "Cityscapes"]:
            root_val = os.path.join(self.args.data_root, 'DensePASS')
            val_list = 'dataset/densepass_list/val.txt'
            target_dataset = WildPASSforAdaptation(pd_dir,
                                                os.path.join(os.path.dirname(pd_dir), 'used_path.txt'),
                                                crop_size=(4096, 1024),
                                                sliding_window=(384, 1024))
            val_dataset = DensePASS13(root_val, val_list, crop_size=(4096, 1024), sliding_window=(384, 1024))

        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        target_loader = DataLoader(target_dataset, batch_size=self.args.batch_size, num_workers=8,
                                drop_last=True,
                                worker_init_fn=lambda x: random.seed(time.time() + x))
                                    
        return target_loader, val_loader
    
    def build_src(self):
        class_names = DATASET_NAME2CLASSES[self.args.dataset]
        num_classes = len(class_names)

        if self.args.dataset == 'Stanford2D3D':
            root_train = os.path.join(self.args.data_root, 'Stanford2d3d_Seg')
            train_list = 'dataset/s2d3d_pin_list/train.txt'
            val_list = 'dataset/s2d3d_pin_list/val.txt'
            train_dataset = StanfordPin8DataSet(root_train, train_list, crop_size=(1024, 1024), set='train')
            val_dataset = StanfordPin8DataSet(root_train, val_list, crop_size=(1024, 1024), set='val')
        elif self.args.dataset == 'SynPASS':
            root_train = os.path.join(self.args.data_root, 'SynPASS')
            train_list = 'dataset/synpass_list/train.txt'
            val_list = 'dataset/synpass_list/val.txt'
            train_dataset = SynPASS13(root_train, train_list, crop_size=(2048, 1024), sliding_window=(128, 1024))
            val_dataset = SynPASS13(root_train, val_list, crop_size=(2048, 1024), sliding_window=(128, 1024))
        elif self.args.dataset == 'Cityscapes':
            root_train = os.path.join(self.args.data_root, 'cityscapes')
            train_list = 'dataset/cityscapes_list/train.txt'
            val_list = 'dataset/cityscapes_list/val.txt'
            train_dataset = Cityscapes13(root_train, train_list, crop_size=(2048, 1024), sliding_window=(128, 1024), set='train')
            val_dataset = Cityscapes13(root_train, val_list, crop_size=(2048, 1024), sliding_window=(128, 1024), set='val')
        else:
            raise ValueError(self.args.dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=8, drop_last=True,
                                  worker_init_fn=lambda x: random.seed(time.time() + x))
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, 
                                num_workers=4, pin_memory=True)    
                                
        return train_loader, val_loader

    def build_criteria(self, device, dataset_name):
        if dataset_name == 'Stanford2D3D':
            crit_s = torch.nn.CrossEntropyLoss(ignore_index=255)
        else:
            weight = torch.Tensor([2.8149, 6.9850, 3.7890, 9.9428, 9.7702, 9.5111,
                                   10.3114, 10.0265, 4.6323, 9.5608, 7.8698, 9.5169, 10.3737]).to(device)
            crit_s = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weight)
        crit_t = torch.nn.CrossEntropyLoss(ignore_index=255)
        return {'s': crit_s, 't': crit_t}

    def write_splits_after_pl(self, save_dir: str, used_paths: List[str]):
        used_path_file = os.path.join(save_dir, 'used_path.txt')
        with open(used_path_file, 'w') as f:
            for p in used_paths:
                f.write(f"{p}\n")

        if os.path.exists(os.path.join(save_dir, 'unused_path.txt')):
            all_path_file = os.path.join(save_dir, 'unused_path.txt')
        else:
            all_path_file = (
                'dataset/densepass_list/train.txt'
                if self.args.dataset != 'Stanford2D3D'
                else 'dataset/s2d3d_pan_list/train.txt'
            )

        with open(all_path_file, 'r') as f1, open(used_path_file, 'r') as f2:
            paths1 = set(f1.read().splitlines())
            paths2 = set(f2.read().splitlines())
        filtered_paths = paths1 - paths2

        train_filtered_file = os.path.join(save_dir, 'unused_path.txt')
        with open(train_filtered_file, 'w') as f_out:
            f_out.write('\n'.join(filtered_paths))