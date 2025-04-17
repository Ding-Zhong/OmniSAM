import argparse
import datetime
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.adaption.cs13_dataset_src import CS13SrcDataSet_sw
from dataset.adaption.dp13_dataset_pd_sw import densepass13TestDataSet
from dataset.adaption.dp13_dataset_pd_sw_it import densepass13TrainDataSet
from dataset.adaption.dp13_train_dataset import densepass13DataSet
from dataset.adaption.sp13_dataset import synpass13DataSet
from dataset.adaption.sp13_dataset_sw import synpass13DataSet_sw
from dataset.adaption.stanford_pan8_dataset_sw import StanfordPan8TestDataSet
from dataset.adaption.stanford_pan8_dataset_sw_it import StanfordPan8DataSet
from dataset.adaption.stanford_pan8_dataset_sw_re import StanfordPan8TrainDataSet
from dataset.adaption.stanford_pin8_dataset import StanfordPin8DataSet
from metrics.compute_iou import fast_hist, per_class_iu
from model import OmniSAM, OmniSAM_adapt
from sam.sam.build_sam import build_sam2
from utils import PrototypicalAdaptation
from utils.tools import *

warnings.filterwarnings('ignore')

DATASET_NAME2CLASSES = {
    "Stanford2D3D": [
        "ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"
    ],
    "Cityscapes": [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "car"
    ],
    "SynPASS": [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "car"
    ],
    "DensePASStrain": [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "car"
    ],
}

COLOR_MAP = {
    0:  (255,   0,   0),
    1:  (  0, 255,   0),
    2:  (  0,   0, 255),
    3:  (255, 255,   0),
    4:  (255,   0, 255),
    5:  (  0, 255, 255),
    6:  (255, 165,   0),
    7:  (128,   0, 128),
    8:  (  0, 255, 127),
    9:  (255, 192, 203),
    10: (139,  69,  19),
    11: (128, 128, 128),
    12: (128, 128,   0),
    13: (75,   0, 130),
    14: (255, 215,   0),
    15: (192, 192, 192),
    16: (0,   128, 128),
    17: (220,  20,  60),
    18: (144, 238, 144),
    255:(255,255,255)
}

def setup_seed(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def adjust_learning_rate_poly(optimizer: optim.Optimizer, current_iter: int,
                              max_iter: int, base_lr: float, power: float = 1.0) -> float:
    lr = base_lr * (1 - (current_iter / max_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def linear_warmup(optimizer: optim.Optimizer, current_iter: int, warmup_iters: int,
                  base_lr: float) -> float:
    lr = base_lr * (current_iter / warmup_iters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def safe_mkdir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def create_overlay(mask: np.ndarray, target_size: tuple, color_map: dict = COLOR_MAP,
                   alpha_value: int = 127) -> Image.Image:
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for label, color in color_map.items():
        indices = (mask == label)
        overlay[indices] = (*color, alpha_value)
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    overlay_img = overlay_img.resize(target_size, Image.NEAREST)
    return overlay_img

def mask_vis(mask_list, name_list, dataset_name, save_path, args):
    """
    根据数据集不同将原图裁剪、保存伪标签以及生成叠加效果图
    """
    if dataset_name == "Stanford2D3D":
        root_file = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/Stanford2d3d_Seg"
        target_size = (3072, 1024)
        for idx, img_name in enumerate(name_list):
            img_path = os.path.join(root_file, img_name)
            image_save_path = os.path.join(save_path, img_name)
            safe_mkdir(os.path.dirname(image_save_path))

            ori_img = Image.open(img_path).convert('RGBA')
            ori_img = ori_img.crop((0, 320, ori_img.width, 1728))
            ori_img.save(image_save_path)

            label_save_path = image_save_path.replace("rgb", "semantic")
            overlay_save_path = image_save_path.replace("rgb", "overlay")

            pseudo_label = Image.fromarray(mask_list[idx].astype(np.uint8), mode='L')
            pseudo_label = pseudo_label.resize(target_size, Image.NEAREST)
            safe_mkdir(os.path.dirname(label_save_path))
            pseudo_label.save(label_save_path)
            print(f"Pseudo label saved to: {label_save_path}")

            overlay_img = create_overlay(mask_list[idx], target_size)
            ori_resized = ori_img.resize(target_size, Image.BICUBIC)
            out_img = Image.alpha_composite(ori_resized, overlay_img)
            safe_mkdir(os.path.dirname(overlay_save_path))
            out_img.save(overlay_save_path)
            print(f"Overlay image saved to: {overlay_save_path}")
    else:
        root_file = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/DensePASStrain"
        save_root_overlap = os.path.join(save_path, "gtFine_overlap")
        save_root_label = os.path.join(save_path, "gtFine")
        safe_mkdir(save_root_overlap)
        safe_mkdir(save_root_label)
        target_size = (2048, 400)
        for idx, img_name in enumerate(name_list):
            img_path = os.path.join(root_file, "leftImg8bit", img_name)
            save_img_path = os.path.join(save_root_overlap, img_name).replace(".jpg", ".png")
            ori_img = Image.open(img_path).convert('RGBA')

            label_name = img_name.replace(".jpg", "labelTrainIds.png")
            label_save_path = os.path.join(save_root_label, label_name)
            pseudo_label = Image.fromarray(mask_list[idx].astype(np.uint8), mode='L')
            safe_mkdir(os.path.dirname(label_save_path))
            pseudo_label.save(label_save_path)
            print(f"Pseudo label saved to: {label_save_path}")

            overlay_img = create_overlay(mask_list[idx], target_size)
            out_img = Image.alpha_composite(ori_img, overlay_img)
            safe_mkdir(os.path.dirname(save_img_path))
            out_img.save(save_img_path)
            print(f"Overlay image saved to: {save_img_path}")

def prepare_pd_labels(model: nn.Module, tgt_data_loader: DataLoader, device: torch.device,
                      num_classes: int, class_names: list, epoch: int, save_path: str,
                      backbone_name: str, num_maskmem: int, best_performance: float,
                      num_pd_labels: int, args):
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    print('----- Start generating pseudo-labels -----')
    coords_list = tgt_data_loader.dataset._coords_list()
    coords_array = np.array(coords_list, dtype=np.int32)
    coords_array = np.tile(coords_array[None, ...], (5, 1, 1))
    path_list = []
    with torch.no_grad():
        for index, batch in enumerate(tgt_data_loader):
            image, _, name = batch
            path_list.extend(name)
            image = image.to(device)
            B = image.shape[0]
            logits_maps = np.zeros((B, 2 * tgt_data_loader.dataset._get_num_maskmem(),
                                    len(class_names), 1024, 1024))
            # 遍历 mask memory 帧
            for frame_idx in range(tgt_data_loader.dataset._get_num_maskmem()):
                frame_img = image[:, frame_idx]
                if args.dataset == "Stanford2D3D":
                    output, _, _ = model(0, frame_img)
                else:
                    output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, frame_idx] = output.cpu().numpy()

            if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "_init_output_dict"):
                model.memory_bank._init_output_dict_source()

            for frame_idx in range(tgt_data_loader.dataset._get_num_maskmem()):
                frame_img = image[:, num_maskmem + frame_idx]
                if args.dataset == "Stanford2D3D":
                    output, _, _ = model(0, frame_img)
                else:
                    output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, num_maskmem + frame_idx] = output.cpu().numpy()

            if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "_init_output_dict"):
                model.memory_bank._init_output_dict_source()

            if args.dataset == "Stanford2D3D":
                merged_pred = sliding_window_prediction(logits_maps, coords_array, orig_size=(1024, 3072))
            else:
                merged_pred = sliding_window_prediction(logits_maps, coords_array, orig_size=(1024, 4096))

            mask_vis(merged_pred, name, args.dataset, save_path, args)

            if index == (num_pd_labels // B - 1):
                val_file = os.path.join(save_path, "val.txt")
                with open(val_file, "w") as f:
                    for p in path_list:
                        f.write(f"{p}\n")
                print(f"Generated {num_pd_labels} pseudo-labels")

                with open(args.filt_file, "r") as f1, open(val_file, "r") as f2:
                    paths1 = set(f1.read().splitlines())
                    paths2 = set(f2.read().splitlines())
                filtered_paths = paths1 - paths2
                train_filtered_file = os.path.join(save_path, "train_filtered.txt")
                with open(train_filtered_file, "w") as f_out:
                    f_out.write("\n".join(filtered_paths))
                break


def validation(model: nn.Module, data_loader: DataLoader, device: torch.device,
               num_classes: int, class_names: list, epoch: int, save_path: str,
               backbone_name: str, num_maskmem: int, best_performance: float,
               best_ckpt_path, args) -> (float, str):
    """
    模型验证过程，计算并打印每个类别的 mIoU，同时保存当前模型和最佳模型权重
    """
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    last_ckpt_path = os.path.join(save_path, f"last_{backbone_name}.pth")
    torch.save(model.state_dict(), last_ckpt_path)

    print('----- Start Validation -----')
    with torch.no_grad():
        for index, batch in enumerate(data_loader):
            if index % 100 == 0:
                print(f"{index} processed")
            image, label, *_ = batch
            image, label = image.to(device), label.to(device)
            for frame_idx in range(num_maskmem):
                frame_img = image[:, frame_idx]
                frame_lbl = label[:, frame_idx]
                if args.dataset == "Stanford2D3D":
                    output, _, _ = model(0, frame_img)
                else:
                    output, _, _ = model(frame_idx, frame_img)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                label_np = frame_lbl.squeeze(0).cpu().numpy()
                hist += fast_hist(label_np.flatten(), pred.flatten(), num_classes)
            if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "_init_output_dict"):
                model.memory_bank._init_output_dict_source()

    mIoUs = per_class_iu(hist)
    for idx, cname in enumerate(class_names):
        print(f"{cname:<15}: {round(mIoUs[idx]*100, 2)}")
    cur_mIoU = round(np.nanmean(mIoUs)*100, 2)
    if cur_mIoU > best_performance:
        best_performance = cur_mIoU
        best_ckpt_path = os.path.join(save_path, f"best_{backbone_name}_iou{best_performance}.pth")
        torch.save(model.state_dict(), best_ckpt_path)
    print(f"Epoch: {epoch}, {backbone_name} val_mIoU: {cur_mIoU}, Best: {best_performance}")
    return best_performance, best_ckpt_path


def train_one_epoch(model: nn.Module, model_s: nn.Module, adaption,
                    source_loader: DataLoader, target_loader: DataLoader,
                    val_loader: DataLoader, criterion_s: nn.Module, criterion_t: nn.Module,
                    optimizer: optim.Optimizer, epoch: int, num_classes, class_names: list,
                    num_epochs: int, device: torch.device, base_lr: float, warmup_iters: int,
                    total_iterations: int, num_maskmem: int, save_path, best_performance,
                    best_ckpt_path, num_pd_labels: int, args):
    running_loss_sup_t = running_loss_sup_s = running_loss_prototype = running_loss = 0.0
    source_loader_it = iter(source_loader)
    target_loader_it = iter(target_loader)

    model.train()
    model_s.eval()

    for it in range(len(target_loader)):
        batch_s = next(source_loader_it)
        batch_t = next(target_loader_it)
        images_s, labels_s, *_ = batch_s
        images_t, labels_t, *_ = batch_t
        images_s, labels_s = images_s.to(device), labels_s.to(device)
        images_t, labels_t = images_t.to(device), labels_t.to(device)
        global_step = epoch * len(source_loader) + it + 1

        for frame_idx in range(num_maskmem):
            frame_img_s = images_s[:, frame_idx, :, :, :]
            frame_lbl_s = labels_s[:, frame_idx, :, :]
            frame_img_t = images_t[:, frame_idx, :, :, :]
            frame_lbl_t = labels_t[:, frame_idx, :, :]

            pred_s, feat_s, feat_s_de = model(frame_idx, frame_img_s, mem_type="source")
            _, feat_s_ori, feat_s_de_ori = model_s(frame_idx, frame_img_s, mem_type="source")
            pred_t, feat_t, feat_t_de = model(frame_idx, frame_img_t, mem_type="target")

            # Prototype Adaptation
            s_pt = adaption.calculate_batch_prototypes(feat_s_de_ori, frame_lbl_s)
            adaption.update_global_prototypes(global_step, frame_idx, s_pt)
            t_pt = adaption.calculate_batch_prototypes(feat_t_de, frame_lbl_t)
            loss_pt = adaption.prototype_loss(frame_idx, t_pt)

            loss_sup_t = criterion_t(pred_t, frame_lbl_t)
            loss_sup_s = criterion_s(pred_s, frame_lbl_s)
            loss = loss_sup_t + loss_sup_s + args.alpha * loss_pt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss_sup_t += loss_sup_t.item()
            running_loss_sup_s += loss_sup_s.item()
            running_loss_prototype += loss_pt.item()
            running_loss += loss.item()

        if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "_init_output_dict"):
            model.memory_bank._init_output_dict_source()
            model.memory_bank._init_output_dict_target()
            model_s.memory_bank._init_output_dict_source()

        avg_loss_sup_t = running_loss_sup_t / num_maskmem
        avg_loss_sup_s = running_loss_sup_s / num_maskmem
        avg_loss_p = running_loss_prototype / num_maskmem
        avg_loss = running_loss / num_maskmem
        print(f"[Epoch {epoch+1}/{num_epochs}] Iter {it+1}/{len(target_loader)}: "
              f"loss={avg_loss:.4f}, loss_sup_t={avg_loss_sup_t:.4f}, "
              f"loss_sup_s={avg_loss_sup_s:.4f}, loss_p={avg_loss_p:.6f}")

        running_loss = running_loss_sup_t = running_loss_sup_s = running_loss_prototype = 0.0
        torch.save(adaption.global_prototypes, f"{args.dataset}_{args.backbone}_prototypes.pt")
        if (it+1) % 25 == 0:
            best_performance, best_ckpt_path = validation(
                model=model, data_loader=val_loader, device=device,
                num_classes=num_classes, class_names=class_names, epoch=epoch,
                save_path=save_path, backbone_name=args.backbone,
                num_maskmem=num_maskmem, best_performance=best_performance,
                best_ckpt_path=best_ckpt_path, args=args
            )

    return best_performance, best_ckpt_path


def train_one_epoch_no_mem(model: nn.Module, model_s, adaption,
                           source_loader: DataLoader, target_loader: DataLoader,
                           val_loader: DataLoader, criterion_s: nn.Module, criterion_t: nn.Module,
                           optimizer: optim.Optimizer, epoch: int, num_classes, class_names: list,
                           num_epochs: int, device: torch.device, base_lr: float, warmup_iters: int,
                           total_iterations: int, num_maskmem: int, save_path, best_performance,
                           num_pd_labels: int, best_ckpt_path, args):
    running_loss_sup_t = running_loss_sup_s = running_loss_prototype = running_loss = 0.0
    source_loader_it = iter(source_loader)
    target_loader_it = iter(target_loader)

    model.train()
    model_s.eval()

    for it in range(len(target_loader)):
        batch_t = next(target_loader_it)
        images_t, labels_t, *_ = batch_t
        images_t, labels_t = images_t.to(device), labels_t.to(device)
        global_step = epoch * len(source_loader) + it + 1
        for frame_idx in range(num_maskmem):
            batch_s = next(source_loader_it)
            images_s, labels_s, *_ = batch_s
            images_s, labels_s = images_s.to(device), labels_s.to(device)
            frame_img_s = images_s
            frame_lbl_s = labels_s
            frame_img_t = images_t[:, frame_idx, :, :, :]
            frame_lbl_t = labels_t[:, frame_idx, :, :]

            pred_s, feat_s, feat_s_de = model(0, frame_img_s, mem_type="source")
            _, feat_s_ori, feat_s_de_ori = model_s(0, frame_img_s, mem_type="source")
            pred_t, feat_t, feat_t_de = model(0, frame_img_t, mem_type="target")

            s_pt = adaption.calculate_batch_prototypes(feat_s_de_ori, frame_lbl_s)
            adaption.update_global_prototypes(global_step, frame_idx, s_pt)
            t_pt = adaption.calculate_batch_prototypes(feat_t_de, frame_lbl_t)
            loss_pt = adaption.prototype_loss(frame_idx, t_pt)

            loss_sup_t = criterion_t(pred_t, frame_lbl_t)
            loss_sup_s = criterion_s(pred_s, frame_lbl_s)
            loss = loss_sup_t + loss_sup_s + args.alpha * loss_pt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss_sup_t += loss_sup_t.item()
            running_loss_sup_s += loss_sup_s.item()
            running_loss_prototype += loss_pt.item()
            running_loss += loss.item()

        if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "_init_output_dict"):
            model.memory_bank._init_output_dict_source()
            model.memory_bank._init_output_dict_target()
            model_s.memory_bank._init_output_dict_source()

        avg_loss_sup_t = running_loss_sup_t / num_maskmem
        avg_loss_sup_s = running_loss_sup_s / num_maskmem
        avg_loss_p = running_loss_prototype / num_maskmem
        avg_loss = running_loss / num_maskmem
        print(f"[Epoch {epoch+1}/{num_epochs}] Iter {it+1}/{len(target_loader)}: "
              f"loss={avg_loss:.4f}, loss_sup_t={avg_loss_sup_t:.4f}, "
              f"loss_sup_s={avg_loss_sup_s:.4f}, loss_p={avg_loss_p:.6f}")
        running_loss = running_loss_sup_t = running_loss_sup_s = running_loss_prototype = 0.0
        torch.save(adaption.global_prototypes, f"{args.dataset}_{args.backbone}_prototypes")
        if (it+1) % 25 == 0:
            best_performance, best_ckpt_path = validation(
                model=model, data_loader=val_loader, device=device,
                num_classes=num_classes, class_names=class_names, epoch=epoch,
                save_path=save_path, backbone_name=args.backbone,
                num_maskmem=num_maskmem, best_performance=best_performance,
                best_ckpt_path=best_ckpt_path, args=args
            )
    return best_performance, best_ckpt_path

def build_tgt_train_dataset_and_loader(args: argparse.Namespace):
    if args.dataset in ["SynPASS", "Cityscapes"]:
        dataset_name = "DensePASStrain"
        class_names = DATASET_NAME2CLASSES[dataset_name]
        num_classes = len(class_names)
        root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/DensePASStrain'
        val_list = args.filt_file
        tgt_train_dataset = densepass13DataSet(root_syn, val_list, crop_size=(4096, 1024),
                                                 sw_setting=(384, 512, 1024))
        tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=5, shuffle=True,
                                      num_workers=4, pin_memory=True)
    elif args.dataset == "Stanford2D3D":
        class_names = DATASET_NAME2CLASSES[args.dataset]
        num_classes = len(class_names)
        root_syn = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/Stanford2d3d_Seg"
        val_list = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/dataset/adaption/s2d3d_pan_list/train.txt"
        tgt_train_dataset = StanfordPan8TrainDataSet(root_syn, val_list, crop_size=(3072, 1024),
                                                     sw_setting=(256, 512, 1024), set="train")
        print(f"Train dataset length: {len(tgt_train_dataset)}")
        tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=5, shuffle=True,
                                      num_workers=4, pin_memory=True)
    return tgt_train_loader, class_names, num_classes


def build_dataset_and_loader(args: argparse.Namespace):
    dataset_name = args.dataset
    if dataset_name not in DATASET_NAME2CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    class_names = DATASET_NAME2CLASSES[dataset_name]
    num_classes = len(class_names)
    if dataset_name == "Stanford2D3D":
        root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/Stanford2d3d_Seg'
        train_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/s2d3d_pin_list/train.txt'
        test_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/s2d3d_pin_list/val.txt'
        train_dataset = StanfordPin8DataSet(root_syn, train_list, crop_size=(1024, 1024), set='train')
        test_dataset = StanfordPin8DataSet(root_syn, test_list, crop_size=(1024, 1024))
        num_maskmem = 9
    elif dataset_name == "SynPASS":
        root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/SynPASS'
        train_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/SynPASS/train.txt'
        test_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/SynPASS/test.txt'
        train_dataset = synpass13DataSet_sw(root_syn, train_list, crop_size=(2048, 1024), 
                                             sw_setting=(128, 512, 1024), set='train')
        test_dataset = synpass13DataSet_sw(root_syn, test_list, crop_size=(2048, 1024), 
                                            sw_setting=(128, 512, 1024))
        num_maskmem = train_dataset._get_num_maskmem()
    elif dataset_name == "Cityscapes":
        root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/cps'
        train_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/cityscapes_list/train.txt'
        test_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/cityscapes_list/val.txt'
        train_dataset = CS13SrcDataSet_sw(root_syn, train_list, crop_size=(2048, 1024),
                                          sw_setting=(128, 512, 1024), set='train')
        test_dataset = CS13SrcDataSet_sw(root_syn, test_list, crop_size=(2048, 1024),
                                         sw_setting=(128, 512, 1024))
        num_maskmem = train_dataset._get_num_maskmem()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, drop_last=True,
                              worker_init_fn=lambda x: random.seed(time.time() + x))
    val_loader = DataLoader(test_dataset, batch_size=5, shuffle=False,
                            num_workers=4, pin_memory=True)

    target_dataset = densepass13TrainDataSet(args.pd_label_save_path,
                                             os.path.join(args.pd_label_save_path, "val.txt"),
                                             crop_size=(4096, 1024), sw_setting=(384, 512, 1024))
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, num_workers=8,
                               drop_last=True,
                               worker_init_fn=lambda x: random.seed(time.time() + x))
    return train_loader, val_loader, target_loader, class_names, num_classes, num_maskmem


def build_target_model(args: argparse.Namespace, num_maskmem: int, device: torch.device):
    sam2_checkpoint = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "/configs/sam2.1/sam2.1_hiera_b+.yaml"

    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
    model = OmniSAM_adapt(sam, num_classes=args.num_classes, num_maskmem=num_maskmem).to(device)

    best_ckpt_path = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/exp_sam2_b+_mem/Cityscapes/best_sam2_b+.pth"
    model.load_state_dict(torch.load(best_ckpt_path))
    return model, best_ckpt_path

def build_source_model(args: argparse.Namespace, num_maskmem: int, device: torch.device):
    sam2_checkpoint = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "/configs/sam2.1/sam2.1_hiera_b+.yaml"

    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
    model = OmniSAM_adapt(sam, num_classes=args.num_classes, num_maskmem=num_maskmem).to(device)

    best_ckpt_path = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/exp_sam2_b+_mem/Cityscapes/best_sam2_b+.pth"
    # load_model_weights(model, best_ckpt_path, remove_module_prefix=True)
    model.load_state_dict(torch.load(best_ckpt_path))

    return model, best_ckpt_path

def main():
    parser = argparse.ArgumentParser(description='PyTorch Single-GPU Training Example')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--backbone', type=str, default='sam2_l',
                        help='sam2_l | sam2_b+ | sam2_s | sam2_t')
    parser.add_argument('--dataset', type=str, default='Stanford2D3D',
                        help='Source domain: Stanford2D3D | SynPASS | Cityscapes')
    args = parser.parse_args()

    setup_seed(args.seed)
    args.cur_epoch = 0
    if args.dataset in ["Cityscapes", "SynPASS"]:
        args.filt_file = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/densepass_list/train.txt'
    elif args.dataset == "Stanford2D3D":
        args.filt_file = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/dataset/adaption/s2d3d_pan_list/train.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tgt_train_loader, class_names, num_classes = build_tgt_train_dataset_and_loader(args)
    args.num_classes = num_classes
    num_maskmem = 9

    model, best_ckpt_path = build_target_model(args, num_maskmem, device)
    model.to(device)
    model2, _ = build_source_model(args, num_maskmem, device)
    model2.to(device)

    cfa_method = PrototypicalAdaptation(num_classes=num_classes, feature_dim=256, num_maskmem=num_maskmem)
    print(f"Using backbone: {args.backbone}, memory bank size: {num_maskmem}")
    print(f"Learning rate: {args.lr}, alpha: {args.alpha}")

    if args.dataset == "Stanford2D3D":
        criterion_sup_s = nn.CrossEntropyLoss(ignore_index=255)
    else:
        weight = torch.Tensor([2.8149, 6.9850, 3.7890, 9.9428, 9.7702, 9.5111,
                                 10.3114, 10.0265, 4.6323, 9.5608, 7.8698, 9.5169, 10.3737])
        weight = weight.to(device)
        criterion_sup_s = nn.CrossEntropyLoss(ignore_index=255, weight=weight)
    criterion_sup_t = nn.CrossEntropyLoss(ignore_index=255)
    criterion_adv = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)

    best_performance = 0.0
    warmup_iters = 400
    iters = 200

    for epoch in range(args.num_epochs):
        args.cur_epoch = epoch
        if args.dataset == "Stanford2D3D":
            pd_label_save_path = f"/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data_syn_pd/Stanford2D3D/{args.backbone}/epoch{epoch%10+1}"
            safe_mkdir(pd_label_save_path)
            for folder in os.listdir(pd_label_save_path):
                if "area" in folder:
                    shutil.rmtree(os.path.join(pd_label_save_path, folder))
                    print(f"Deleted: {folder}")
            val_txt_path = os.path.join(pd_label_save_path, "val.txt")
            if os.path.exists(val_txt_path):
                os.remove(val_txt_path)
                print(f"Deleted file: {val_txt_path}")
        else:
            if epoch > 0:
                args.filt_file = f"/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data_syn_pd/DensePASS/{args.backbone}/{args.dataset}/epoch{(epoch % 10)}/train_filtered.txt"
            pd_label_save_path = f"/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data_syn_pd/DensePASS/{args.backbone}/{args.dataset}/epoch{(epoch % 10) + 1}"
            safe_mkdir(pd_label_save_path)
            for dir_name in ["gtFine", "gtFine_overlap"]:
                dir_path = os.path.join(pd_label_save_path, dir_name)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    print(f"Deleted: {dir_path}")
            for txt in ["val.txt", "train_filtered.txt"]:
                txt_path = os.path.join(pd_label_save_path, txt)
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                    print(f"Deleted file: {txt_path}")
        args.pd_label_save_path = pd_label_save_path
        save_path = pd_label_save_path

        num_pd_labels = warmup_iters if epoch == 0 else iters
        prepare_pd_labels(model, tgt_train_loader, device, num_classes, class_names,
                          epoch, pd_label_save_path, args.backbone, num_maskmem,
                          best_performance, num_pd_labels, args)

        source_loader, val_loader, target_loader, class_names, num_classes, num_maskmem = build_dataset_and_loader(args)
        total_iterations = args.num_epochs * len(target_loader)
        print("------Start adaptation-----")
        if args.dataset == "Stanford2D3D":
            best_performance, best_ckpt_path = train_one_epoch_no_mem(
                model=model, model_s=model2, adaption=cfa_method, source_loader=source_loader,
                target_loader=target_loader, val_loader=val_loader, criterion_s=criterion_sup_s,
                criterion_t=criterion_sup_t, optimizer=optimizer, epoch=epoch, num_classes=num_classes,
                class_names=class_names, num_epochs=args.num_epochs, device=device, base_lr=args.lr,
                warmup_iters=warmup_iters, total_iterations=total_iterations, num_maskmem=num_maskmem,
                save_path=save_path, best_performance=best_performance, num_pd_labels=num_pd_labels,
                best_ckpt_path=best_ckpt_path, args=args)
        else:
            best_performance, best_ckpt_path = train_one_epoch(
                model=model, model_s=model2, adaption=cfa_method, source_loader=source_loader,
                target_loader=target_loader, val_loader=val_loader, criterion_s=criterion_sup_s,
                criterion_t=criterion_sup_t, optimizer=optimizer, epoch=epoch, num_classes=num_classes,
                class_names=class_names, num_epochs=args.num_epochs, device=device, base_lr=args.lr,
                warmup_iters=warmup_iters, total_iterations=total_iterations, num_maskmem=num_maskmem,
                save_path=save_path, best_performance=best_performance, best_ckpt_path=best_ckpt_path,
                num_pd_labels=num_pd_labels, args=args)

        if epoch % 5 == 0:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    print("Training Finished.")


if __name__ == "__main__":
    print("File name:", __file__)
    main()
