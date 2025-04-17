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

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from metrics.compute_iou import fast_hist, per_class_iu
from dataset.adaption.stanford_pin8_dataset_sw import StanfordPin8DataSet
from dataset.adaption.stanford_pan8_dataset_sw_re import StanfordPan8TestDataSet
from dataset.adaption.sp13_dataset_sw import synpass13DataSet_sw
from dataset.adaption.dp13_dataset_pd_sw_re import densepass13TestDataSet, densepass13TestDataSet_new
from dataset.adaption.dp13_train_dataset import densepass13DataSet
from dataset.adaption.cs13_dataset_src import CS13SrcDataSetval_sw
from sam.sam.build_sam import build_sam2
from model import OmniSAM, OmniSAM_adapt
from utils.tools import *
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


def adjust_learning_rate_poly(optimizer: optim.Optimizer, current_iter: int, max_iter: int,
                              base_lr: float, power: float = 1.0) -> float:
    """Adjust learning rate using poly schedule."""
    lr = base_lr * (1 - (current_iter / max_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def linear_warmup(optimizer: optim.Optimizer, current_iter: int, warmup_iters: int,
                  base_lr: float) -> float:
    """Perform linear warmup of learning rate."""
    lr = base_lr * (current_iter / warmup_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def mask_vis(mask, names, dataset_name, save_root, args):
    """
    Visualize and save the pseudo-label masks.
    
    Depending on the dataset, the image and label saving steps vary.
    """
    color_map = {
        0: (128, 64, 128), 1: (244, 35, 232), 2: (70, 70, 70),
        3: (102, 102, 156), 4: (190, 153, 153), 5: (153, 153, 153),
        6: (250, 170, 30), 7: (220, 220, 0), 8: (107, 142, 35),
        9: (152, 251, 152), 10: (70, 130, 180), 11: (220, 20, 60),
        12: (0, 0, 142), 13: (75, 0, 130), 14: (255, 215, 0),
        15: (192, 192, 192), 16: (0, 128, 128), 17: (220, 20, 60),
        18: (144, 238, 144)
    }
    
    if dataset_name == "Stanford2D3D":
        root_file = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/Stanford2d3d_Seg"
        for idx, img_name in enumerate(names):
            img_path = os.path.join(root_file, img_name)
            image_save_path = os.path.join(save_root, img_name)
            
            # Open and crop the original image
            ori_img = Image.open(img_path).convert('RGBA')
            width, height = ori_img.size
            ori_img = ori_img.crop((0, 320, width, 1728))
            os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
            ori_img.save(image_save_path)
            
            label_save_path = image_save_path.replace("rgb", "semantic")
            overlay_img_save_path = image_save_path.replace("rgb", "overlay")
            
            pseudo_label = Image.fromarray(mask[idx].astype(np.uint8), mode='L')
            os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
            pseudo_label = pseudo_label.resize((3072, 1024), Image.NEAREST)
            pseudo_label.save(label_save_path)
            print(f"Pseudo label saved to: {label_save_path}")

            # Create and composite overlay image
            overlay_data = create_overlay(mask[idx], color_map, alpha=127)
            overlay_img = Image.fromarray(overlay_data, mode="RGBA")
            ori_img = ori_img.resize((3072, 1024), Image.BICUBIC)
            out_img = Image.alpha_composite(ori_img, overlay_img)
            os.makedirs(os.path.dirname(overlay_img_save_path), exist_ok=True)
            out_img.save(overlay_img_save_path)
            print(f"Overlay mask saved to: {overlay_img_save_path}")
    else:
        # Set root file based on dataset
        root_file = os.path.join("/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data", dataset_name)
        save_root_label = os.path.join(
            "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/pseudo_labels",
            args.backbone, args.dataset, "gtFine", "val"
        )
        os.makedirs(save_root_label, exist_ok=True)
        for idx, img_name in enumerate(names):
            if dataset_name == "SynPASS":
                img_path = os.path.join(root_file, img_name)
                save_path = os.path.join(save_root, img_name[-10:-4] + ".png")
                ori_img = Image.open(img_path).convert('RGBA')
            elif dataset_name == "DensePASS":
                img_path = os.path.join(root_file, "leftImg8bit", "val", img_name)
                save_path = os.path.join(save_root, img_name)
                ori_img = Image.open(img_path).convert('RGBA')
                label_name = img_name.replace(".png", "labelTrainIds.png")
                label_save_path = os.path.join(save_root_label, label_name)

            pseudo_label = Image.fromarray(mask[idx].astype(np.uint8), mode='L')
            os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
            pseudo_label.save(label_save_path)
            print(f"Pseudo label saved to: {label_save_path}")

            overlay_data = create_overlay(mask[idx], color_map, alpha=127)
            # Resize overlay differently for DensePASS
            if dataset_name == "DensePASS":
                overlay_img = Image.fromarray(overlay_data, mode="RGBA").resize((2048, 400), Image.NEAREST)
            else:
                overlay_img = Image.fromarray(overlay_data, mode="RGBA")
                ori_img = ori_img.resize((2048, 1024), Image.BICUBIC)
            out_img = Image.alpha_composite(ori_img, overlay_img)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out_img.save(save_path)
            print(f"Overlay mask saved to: {save_path}")


def validation(model: nn.Module, data_loader: DataLoader, device: torch.device,
               num_classes: int, class_names: list, backbone_name: str,
               num_maskmem: int, dataset_name: str, args) -> float:
    """
    Run validation using sliding window predictions.
    """
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    print('----- Start Validation -----')

    # Obtain and tile coordinate list
    coords_list = data_loader.dataset._coords_list()
    coords_array = np.array(coords_list, dtype=np.int32)
    coords_array = np.tile(coords_array[None, ...], (args.batch_size, 1, 1))
    
    # Cache the number of mask memories to avoid multiple calls.
    num_mem = data_loader.dataset._get_num_maskmem()

    with torch.no_grad():
        for index, batch in enumerate(data_loader):
            image, label, _, names, ori_label = batch
            image = image.to(device)
            label = label.to(device)
            ori_label = ori_label.to(device).squeeze(0).cpu().numpy()
            B = image.shape[0]

            # Initialize logits_maps for both groups of mask memories
            logits_maps = np.zeros((B, 2 * num_mem, len(class_names), 1024, 1024))

            # Process first set of frames
            for frame_idx in range(num_mem):
                frame_img = image[:, frame_idx]
                frame_label = label[:, frame_idx]
                if dataset_name == "Stanford2D3D":
                    output, feat, feat_de = model(0, frame_img)
                else:
                    output, feat, feat_de = model(frame_idx, frame_img)
                logits_maps[:, frame_idx] = output.cpu().numpy()

            if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "_init_output_dict"):
                model.memory_bank._init_output_dict_source()

            # Process second set of frames
            for frame_idx in range(num_mem):
                frame_img = image[:, num_mem + frame_idx]
                frame_label = label[:, num_mem + frame_idx]
                if dataset_name == "Stanford2D3D":
                    output, _, _ = model(0, frame_img)
                else:
                    output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, num_mem + frame_idx] = output.cpu().numpy()

            if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "_init_output_dict"):
                model.memory_bank._init_output_dict_source()

            # Merge predictions based on dataset
            if dataset_name == "DensePASS":
                merged_pred = sliding_window_prediction(
                    logits_maps, coords_array, orig_size=(1024, 4096))

                pred_resized = np.zeros((B, 400, 2048))
                for i in range(B):
                    pred_resized[i] = cv2.resize(merged_pred[i], (2048, 400),
                                                    interpolation=cv2.INTER_NEAREST)
                hist_np = fast_hist(ori_label.flatten(), pred_resized.flatten().astype(int), num_classes)

                hist += hist_np

            elif dataset_name == "Cityscapes":
                merged_pred = sliding_window_prediction(
                    logits_maps, coords_array, orig_size=(1024, 2048))
                hist_np = fast_hist(ori_label.flatten(), merged_pred.flatten().astype(int), num_classes)
                hist += hist_np

            elif dataset_name == "Stanford2D3D":
                merged_pred = sliding_window_prediction(
                    logits_maps, coords_array, orig_size=(1024, 3072))
                hist_np = fast_hist(ori_label.flatten(), merged_pred.flatten().astype(int), num_classes)
                hist += hist_np

            save_root = os.path.join(
                "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/output_masks",
                args.backbone, dataset_name)
            os.makedirs(save_root, exist_ok=True)
            # Uncomment the following line if you wish to save visualizations:
            # mask_vis(merged_pred, names, dataset_name, save_root=save_root, args=args)
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
        val_dataset = StanfordPan8TestDataSet(root_syn, val_list, crop_size=(syn_w, syn_h),
                                              sw_setting=(sw_w_stride, sw_h_stride, sw_size))
    elif dataset_name == "SynPASS":
        root_syn = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/SynPASS"
        val_list = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/SynPASS/val.txt"
        syn_w, syn_h = 2048, 1024
        sw_w_stride, sw_h_stride, sw_size = 128, 512, 1024  
        val_dataset = synpass13DataSet_sw(root_syn, val_list, crop_size=(syn_w, syn_h),
                                          sw_setting=(sw_w_stride, sw_h_stride, sw_size))
    elif dataset_name == "Cityscapes":
        root_syn = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/cps"
        val_list = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/cityscapes_list/val.txt"
        syn_w, syn_h = 2048, 1024
        sw_w_stride, sw_h_stride, sw_size = 128, 512, 1024  
        val_dataset = CS13SrcDataSetval_sw(root_syn, val_list, crop_size=(syn_w, syn_h),
                                           sw_setting=(sw_w_stride, sw_h_stride, sw_size))
    elif dataset_name == "DensePASS":
        root_syn = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/DensePASS"
        val_list = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/densepass_list/val.txt"
        syn_w, syn_h = 4096, 1024
        sw_w_stride, sw_h_stride, sw_size = 256, 512, 1024
        val_dataset = densepass13TestDataSet(root_syn, val_list, crop_size=(syn_w, syn_h),
                                             sw_setting=(sw_w_stride, sw_h_stride, sw_size))

    else:
        raise ValueError("Unknown dataset choice.")

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    return val_loader, class_names, num_classes

def build_sam_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """
    Build and load the SAM model with the specified backbone and weights.
    """
    sam2_checkpoint = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "/configs/sam2.1/sam2.1_hiera_b+.yaml"

    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
    model = OmniSAM_adapt(sam, num_classes=args.num_classes, num_maskmem=args.num_maskmem).to(device)

    best_ckpt_path = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/exp_sam2_b+_mem/Cityscapes/best_sam2_b+.pth"
    model.load_state_dict(torch.load(best_ckpt_path))
    
    return model


def main():
    parser = argparse.ArgumentParser(description='PyTorch Single-GPU Training Example')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--backbone', type=str, default='sam2_l',
                        help='Backbone model: sam2_l | sam2_b+ | sam2_s | sam2_t')
    parser.add_argument('--dataset', type=str, default='Stanford2D3D',
                        help='Dataset: Stanford2D3D | SynPASS | DensePASS | DensePASStrain')
    parser.add_argument('--use_mem_bank', action='store_true',
                        help='Whether to use memory bank')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='Input batch size for training (default: 1)')
    args = parser.parse_args()
    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, class_names, num_classes = build_dataset_and_loader(args)
    args.num_classes = num_classes  # Make number of classes available for model building

    args.num_maskmem = 9 if args.use_mem_bank else 0
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
