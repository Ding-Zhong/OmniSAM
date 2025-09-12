import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import time
import datetime
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import cv2

from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.metrics import fast_hist, per_class_iu
from dataset.DensePASS import DensePASS13
from sam.sam.build_sam import build_sam2
from model.model import OmniSAM, OmniSAM_adapt
from utils.tools import *
import time
# Mapping dataset names to their classes
DATASET_NAME2CLASSES = {
    "Stanford2D3D": ["ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"],
    "SynPASS": ["road", "sidewalk", "building", "wall", "fence", "pole",
                "traffic light", "traffic sign", "vegetation", "terrain",
                "sky", "person", "car"],
    "Cityscapes": ["road", "sidewalk", "building", "wall", "fence", "pole",
                   "traffic light", "traffic sign", "vegetation", "terrain",
                   "sky", "person", "car"],
    "DensePASS": ["road", "sidewalk", "building", "wall", "fence", "pole",
                  "traffic light", "traffic sign", "vegetation", "terrain",
                  "sky", "person", "car"]
}

def setup_seed(seed: int = 1234):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Set deterministic mode for cuDNN for a trade-off between determinism and performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def validation(model: nn.Module, data_loader: DataLoader, device: torch.device,
               num_classes: int, class_names: list, backbone_name: str,
               num_maskmem: int, dataset_name: str, args) -> float:
    """
    Run validation using sliding window predictions.
    """
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    print('----- Start Validation -----')

    num_frames = data_loader.dataset._get_frame_cnt()

    with torch.no_grad():
        for index, batch in enumerate(data_loader):
            image, label, _, names, ori_label = batch
            image = image.to(device)
            label = label.to(device)
            ori_label = ori_label.to(device).squeeze(0).cpu().numpy()
            B = image.shape[0]
            H, W = image.shape[-2:]
            # Initialize logits_maps for both groups of mask memories
            logits_maps = np.zeros((B, 2 * num_frames, len(class_names), H, W))

            # Process first set of frames
            for frame_idx in range(num_frames):
                frame_img = image[:, frame_idx]
                frame_label = label[:, frame_idx]
                if dataset_name == "Stanford2D3D":
                    output, feat, feat_de = model(0, frame_img)
                else:
                    output, feat, feat_de = model(frame_idx, frame_img)
                logits_maps[:, frame_idx] = output.cpu().numpy()


            model.memory_bank._init_output_dict_source()

            # Process second set of frames
            for frame_idx in range(num_frames):
                frame_img = image[:, num_frames + frame_idx]
                frame_label = label[:, num_frames + frame_idx]
                if dataset_name == "Stanford2D3D":
                    output, _, _ = model(0, frame_img)
                else:
                    output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, num_frames + frame_idx] = output.cpu().numpy()

            model.memory_bank._init_output_dict_source()

            # Merge predictions based on dataset
            if dataset_name == "DensePASS":
                merged_pred = merge_width_scatter_np(
                    logits_np=logits_maps,           # (N,B,C,H,W_patch)
                    orig_size=(1024, 4096),
                    W_patch=W,
                    stride_w=384,
                    passes=2,
                    device="cuda",
                    return_probs=False
                )
                pred_resized = np.zeros((B, 400, 2048))
                for i in range(B):
                    pred_resized[i] = cv2.resize(merged_pred[i], (2048, 400),
                                                    interpolation=cv2.INTER_NEAREST)
                hist_np = fast_hist(ori_label.flatten(), pred_resized.flatten().astype(int), num_classes)

                hist += hist_np
            save_root = os.path.join(
                "/scratch/chaijy_root/chaijy2/dingdd/OmniSAM/output_masks",
                args.backbone, dataset_name)
            print(f'{(index+1)*B} items processed')
            mIoUs = per_class_iu(hist)
            cur_mIoU = round(np.nanmean(mIoUs) * 100, 2)
            print(f'{backbone_name} val mIoU: {cur_mIoU}')

    mIoUs = per_class_iu(hist)
    for idx, cname in enumerate(class_names):
        print(f'===> {cname:<15}:\t {round(mIoUs[idx] * 100, 2)}')
    cur_mIoU = round(np.nanmean(mIoUs) * 100, 2)
    print(f'{backbone_name} val mIoU: {cur_mIoU}')
    return cur_mIoU

def build_dataset_and_loader(args: argparse.Namespace):
    """
    Build validation dataset and dataloader based on the chosen dataset.
    """
    dataset_name = args.dataset
    if dataset_name not in DATASET_NAME2CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    class_names = DATASET_NAME2CLASSES[dataset_name]
    num_classes = len(class_names)

    if dataset_name == "Stanford2D3D":
        root_syn = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/Stanford2d3d_Seg"
        val_list = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/s2d3d_pan_list/val.txt"
        syn_w, syn_h = 3072, 1024
        sw_w_stride, sw_h_stride, sw_size = 256, 512, 1024 
        assert (syn_w - sw_size) % sw_w_stride == 0 and  sw_w_stride <= sw_size
        # val_dataset = StanfordPan8TestDataSet(root_syn, val_list, crop_size=(syn_w, syn_h),
        #                                       sw_setting=(sw_w_stride, sw_h_stride, sw_size))
    elif dataset_name == "DensePASS":
        root_syn = "data/DensePASS"
        val_list = "dataset/densepass_list/val.txt"
        syn_w, syn_h = 4096, 1024
        sw_w_stride, sw_size = 384, 1024
        assert (syn_w - sw_size) % sw_w_stride == 0 and  sw_w_stride <= sw_size
        val_dataset = DensePASS13(root_syn, val_list, crop_size=(syn_w, syn_h), sliding_window=(sw_w_stride, sw_size), passes=2)

    else:
        raise ValueError("Unknown dataset choice.")

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    return val_loader, class_names, num_classes

def build_sam_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """
    Build and load the SAM model with the specified backbone and weights.
    """
    sam2_checkpoint = "sam/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "/configs/sam2.1/sam2.1_hiera_s.yaml"

    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
    model = OmniSAM_adapt(sam, num_classes=args.num_classes, num_maskmem=args.num_maskmem).to(device)

    best_ckpt_path = "" # the checkpoint file path

    state_dict = torch.load(best_ckpt_path, map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch Single-GPU Training Example')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--backbone', type=str, default='sam2_s',
                        help='Backbone model: sam2_l | sam2_b+ | sam2_s | sam2_t')
    parser.add_argument('--dataset', type=str, default='DensePASS',
                        help='Dataset: Stanford2D3D | DensePASS')
    parser.add_argument('--use_mem_bank', action='store_true',
                        help='Whether to use memory bank')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 1)')
    args = parser.parse_args()
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, class_names, num_classes = build_dataset_and_loader(args)
    args.num_classes = num_classes  # Make number of classes available for model building

    args.num_maskmem = 3 if args.use_mem_bank else 0
    print(f"Using backbone: {args.backbone}, Use memory bank: {args.use_mem_bank}, num_maskmem: {args.num_maskmem}")

    model = build_sam_model(args, device).to(device)
    
    validation(
        model=model,
        data_loader=val_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        backbone_name=args.backbone,
        num_maskmem=args.num_maskmem,
        dataset_name=args.dataset,
        args=args
    )

    print("Validation Finished.")


if __name__ == "__main__":
    print("File name:", __file__)
    main()
