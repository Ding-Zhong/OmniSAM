import argparse
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Custom modules (dataset, models, etc.)
from metrics.compute_iou import fast_hist, per_class_iu
from dataset.adaption.cs13_dataset_src import CS13SrcDataSet_sw
from dataset.adaption.cs_dataset_src import CS19SrcDataSet_sw
from dataset.adaption.dp13_dataset_pd_sw_re import densepass13ValDataSet
from dataset.adaption.dp13_train_dataset import densepass13DataSet
from dataset.adaption.sp13_dataset import synpass13DataSet
from dataset.adaption.sp13_dataset_sw import synpass13DataSet_sw
from dataset.adaption.stanford_pin8_dataset import StanfordPin8DataSet
from sam.sam.build_sam import build_sam2
from model import OmniSAM

# Suppress warnings
warnings.filterwarnings('ignore')

DATASET_NAME2CLASSES = {
    "Stanford2D3D": [
        "ceiling", "chair", "door", "floor", "sofa", "table", "wall", "window"
    ],
    "SynPASS": [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "car"
    ],
    "Cityscapes": [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "car"
    ],
    "Cityscapes19": [
        "road", "sidewalk", "building", "wall", "fence", "pole",
        "traffic light", "traffic sign", "vegetation", "terrain",
        "sky", "person", "Rider", "car", "Truck", "Bus", "Train",
        "Motor Cycle", "Bicycle"
    ],
}


def setup_seed(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def adjust_learning_rate_poly(optimizer: optim.Optimizer,
                              current_iter: int,
                              max_iter: int,
                              base_lr: float,
                              power: float = 1.0) -> float:
    """
    Adjust the learning rate using a polynomial decay schedule.
    """
    lr = base_lr * (1 - (current_iter / max_iter)) ** power
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def linear_warmup(optimizer: optim.Optimizer,
                  current_iter: int,
                  warmup_iters: int,
                  base_lr: float) -> float:
    """
    Adjust the learning rate using a linear warmup schedule.
    """
    lr = base_lr * (current_iter / warmup_iters)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def validation(model: nn.Module,
               data_loader: DataLoader,
               device: torch.device,
               num_classes: int,
               class_names: list,
               writer: SummaryWriter,
               epoch: int,
               save_path: str,
               backbone_name: str,
               num_maskmem: int,
               best_performance: float,
               rank,
               world_size,
               args) -> float:
    """
    Evaluate the model on the validation set and save the best performing checkpoint.
    """
    model.eval()
    # Use a torch tensor for local histogram accumulation in a distributed manner.
    local_hist = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    last_ckpt_path = os.path.join(save_path, f"last_{backbone_name}.pth")

    if rank == 0:
        torch.save(model.state_dict(), last_ckpt_path)
        print("----- Start Validation -----")

    torch.cuda.synchronize(device)
    start_time = time.time()

    with torch.no_grad():
        for index, batch in enumerate(data_loader):
            if rank == 0 and index % 100 == 0:
                print(f"{index} processed")

            image, label, _, _ = batch
            image, label = image.to(device), label.to(device)

            if args.dataset == "Stanford2D3D":
                output = model(0, image)
                pred = torch.argmax(output, dim=1)

                pred_np = pred.squeeze(0).cpu().numpy()  # [H, W]
                label_np = label.squeeze(0).cpu().numpy()

                hist_np = fast_hist(label_np.flatten(), pred_np.flatten(), num_classes)
                hist_torch = torch.from_numpy(hist_np).to(device, dtype=torch.long)
                local_hist += hist_torch

            else:
                for frame_idx in range(num_maskmem):
                    frame_img = image[:, frame_idx]  # [B, 3, H, W]
                    frame_label = label[:, frame_idx]  # [B, H, W]

                    output = model(frame_idx, frame_img)  # [B, num_classes, H, W]
                    pred = torch.argmax(output, dim=1)

                    pred_np = pred.squeeze(0).cpu().numpy()  # [H, W]
                    label_np = frame_label.squeeze(0).cpu().numpy()

                    hist_np = fast_hist(label_np.flatten(), pred_np.flatten(), num_classes)
                    hist_torch = torch.from_numpy(hist_np).to(device, dtype=torch.long)
                    local_hist += hist_torch

            # Reset memory bank output dictionary if applicable.
            if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "_init_output_dict"):
                model.memory_bank._init_output_dict()

    dist.all_reduce(local_hist, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    end_time = time.time()
    val_time = end_time - start_time

    if rank == 0:
        global_hist_np = local_hist.cpu().numpy()
        mIoUs = per_class_iu(global_hist_np)
        for idx, class_name in enumerate(class_names):
            class_mIoU = round(mIoUs[idx] * 100, 2)
            print(f"===> {class_name:<15}:\t {class_mIoU}")

        cur_mIoU = round(np.nanmean(mIoUs) * 100, 2)
        writer.add_scalar(f"[{backbone_name}] val_mIOU", cur_mIoU, epoch)

        if cur_mIoU > best_performance:
            best_performance = cur_mIoU
            best_ckpt_path = os.path.join(save_path, f"best_{backbone_name}.pth")
            torch.save(model.state_dict(), best_ckpt_path)

        print(f"Epoch: {epoch+1}, {backbone_name} val_mIoU: {cur_mIoU}, Best: {best_performance}")

    return best_performance


def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    num_epochs: int,
                    writer: SummaryWriter,
                    device: torch.device,
                    base_lr: float,
                    warmup_iters: int,
                    total_iterations: int,
                    num_maskmem: int,
                    use_mem_bank: bool,
                    rank,
                    world_size,
                    args):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for it, (images, labels, _, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        if args.dataset == "Stanford2D3D":
            pred = model(0, images)
            loss_sup = criterion(pred, labels)

            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()

            if rank == 0:
                global_step = epoch * len(train_loader) + it
                writer.add_scalar("Sup Loss", loss_sup.item(), global_step)

            running_loss += loss_sup.item()

        else:
            for frame_idx in range(num_maskmem):
                frame_img = images[:, frame_idx, :, :, :]  # [B, C, H, W]
                frame_lbl = labels[:, frame_idx, :, :]      # [B, H, W]

                pred = model(frame_idx, frame_img)
                loss_sup = criterion(pred, frame_lbl)

                optimizer.zero_grad()
                loss_sup.backward()
                optimizer.step()

                if rank == 0:
                    global_step = (epoch * len(train_loader) * num_maskmem +
                                   it * num_maskmem + frame_idx)
                    writer.add_scalar("Sup Loss", loss_sup.item(), global_step)

                running_loss += loss_sup.item()

        if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "_init_output_dict"):
            model.memory_bank._init_output_dict_source()

        current_iter = epoch * len(train_loader) + it
        if current_iter < warmup_iters:
            lr_ = linear_warmup(optimizer, current_iter, warmup_iters, base_lr)
        else:
            lr_ = adjust_learning_rate_poly(
                optimizer=optimizer,
                current_iter=current_iter - warmup_iters,
                max_iter=total_iterations - warmup_iters,
                base_lr=base_lr,
            )

        if it % 10 == 0 and rank == 0:
            if args.dataset == "Stanford2D3D":
                avg_loss = running_loss / 10 if it > 0 else running_loss
                print(f"[Epoch {epoch+1}/{num_epochs}] Iter {it}/{len(train_loader)}: "
                      f"Loss={avg_loss:.4f}, LR={lr_:.8f}")
            else:
                avg_loss = running_loss / (10 * num_maskmem) if it > 0 else running_loss / num_maskmem
                print(f"[Epoch {epoch+1}/{num_epochs}] Iter {it}/{len(train_loader)}: "
                      f"Loss={avg_loss:.4f}, LR={lr_:.8f}")
            running_loss = 0.0


def build_dataset_and_loader(args: argparse.Namespace):
    """
    Construct training, validation, and test datasets along with DataLoaders based on input arguments.
    """
    dataset_name = args.dataset
    if dataset_name not in DATASET_NAME2CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    class_names = DATASET_NAME2CLASSES[dataset_name]
    num_classes = len(class_names)

    if dataset_name == "Stanford2D3D":
        root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/Stanford2d3d_Seg'
        train_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/s2d3d_pin_list/train.txt'
        val_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/s2d3d_pin_list/trainval.txt'
        test_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/s2d3d_pin_list/val.txt'

        syn_h, syn_w = 1024, 1024

        train_dataset = StanfordPin8DataSet(root_syn, train_list,
                                            crop_size=(syn_w, syn_h),
                                            set='train')
        val_dataset = StanfordPin8DataSet(root_syn, val_list,
                                          crop_size=(syn_w, syn_h))
        test_dataset = StanfordPin8DataSet(root_syn, test_list,
                                           crop_size=(syn_w, syn_h))
        num_maskmem = 9

    elif dataset_name == "SynPASS":
        root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/SynPASS'
        train_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/SynPASS/train.txt'
        val_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/SynPASS/val.txt'
        test_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/SynPASS/test.txt'

        syn_w, syn_h = 2048, 1024
        sw_w_stride, sw_h_stride, sw_size = 128, 512, 1024

        train_dataset = synpass13DataSet_sw(root_syn, train_list,
                                            crop_size=(syn_w, syn_h),
                                            sw_setting=(sw_w_stride, sw_h_stride, sw_size),
                                            set='train')
        val_dataset = synpass13DataSet_sw(root_syn, val_list,
                                          crop_size=(syn_w, syn_h),
                                          sw_setting=(sw_w_stride, sw_h_stride, sw_size))

        root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/DensePASS'
        val_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/densepass_list/val.txt'

        syn_w, syn_h = 4096, 1024
        sw_w_stride, sw_h_stride, sw_size = 384, 512, 1024
        test_dataset = densepass13ValDataSet(root_syn, val_list,
                                             crop_size=(syn_w, syn_h),
                                             sw_setting=(sw_w_stride, sw_h_stride, sw_size))

        num_maskmem = train_dataset._get_num_maskmem()

    elif dataset_name == "Cityscapes":
        root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/cps'
        train_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/cityscapes_list/train.txt'
        val_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/cityscapes_list/val.txt'
        test_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/cityscapes_list/val.txt'

        syn_w, syn_h = 2048, 1024
        sw_w_stride, sw_h_stride, sw_size = 128, 512, 1024

        train_dataset = CS13SrcDataSet_sw(root_syn, train_list,
                                          crop_size=(syn_w, syn_h),
                                          sw_setting=(sw_w_stride, sw_h_stride, sw_size),
                                          set='train')
        val_dataset = CS13SrcDataSet_sw(root_syn, val_list,
                                        crop_size=(syn_w, syn_h),
                                        sw_setting=(sw_w_stride, sw_h_stride, sw_size))
        test_dataset = CS13SrcDataSet_sw(root_syn, test_list,
                                         crop_size=(syn_w, syn_h),
                                         sw_setting=(sw_w_stride, sw_h_stride, sw_size))
        num_maskmem = train_dataset._get_num_maskmem()

    elif dataset_name == "Cityscapes19":
        root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/cps'
        train_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/cityscapes_list/train.txt'
        val_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/cityscapes_list/val.txt'
        test_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/cityscapes_list/val.txt'

        syn_w, syn_h = 2048, 1024
        sw_w_stride, sw_h_stride, sw_size = 128, 512, 1024

        train_dataset = CS19SrcDataSet_sw(root_syn, train_list,
                                          crop_size=(syn_w, syn_h),
                                          sw_setting=(sw_w_stride, sw_h_stride, sw_size),
                                          set='train')
        val_dataset = CS19SrcDataSet_sw(root_syn, val_list,
                                        crop_size=(syn_w, syn_h),
                                        sw_setting=(sw_w_stride, sw_h_stride, sw_size))
        test_dataset = CS19SrcDataSet_sw(root_syn, test_list,
                                         crop_size=(syn_w, syn_h),
                                         sw_setting=(sw_w_stride, sw_h_stride, sw_size))
        num_maskmem = train_dataset._get_num_maskmem()
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=12,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, val_loader, test_loader, class_names, num_classes, num_maskmem


def build_sam_model(args: argparse.Namespace, num_maskmem: int,
                    device: torch.device, local_rank) -> nn.Module:
    """
    Build and load a SAM model based on the provided backbone parameter.
    """
    if args.backbone == "sam2_l":
        sam2_checkpoint = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/sam/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "/configs/sam2.1/sam2.1_hiera_l.yaml"
    elif args.backbone == "sam2_b+":
        sam2_checkpoint = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/sam/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = "/configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif args.backbone == "sam2_s":
        sam2_checkpoint = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/sam/checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "/configs/sam2.1/sam2.1_hiera_s.yaml"
    elif args.backbone == "sam2_t":
        sam2_checkpoint = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/sam/checkpoints/sam2.1_hiera_tiny.pt"
        model_cfg = "/configs/sam2.1/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown backbone choice.")

    sam = build_sam2(model_cfg, sam2_checkpoint, device=device)
    model = OmniSAM(sam, num_classes=args.num_classes, num_maskmem=num_maskmem).to(device)

    if args.load_ckpt:
        if args.dataset == "Cityscapes19":
            if args.backbone == "sam2_l":
                ckpt_path = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/exp_sam2_l_mem_lr_6e-5_ddp/Cityscapes19/best_sam2_l.pth"
            elif args.backbone == "sam2_b+":
                ckpt_path = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/exp_sam2_b+_mem_lr_6e-5_ddp/Cityscapes19/best_sam2_b+.pth"
            elif args.backbone == "sam2_s":
                ckpt_path = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/exp_sam2_s_mem_adaption_MPA_weight_lr_1e-4/alpha_1.0_lamda_1.0_beta_1.0/Cityscapes/best_sam2_s.pth"
            elif args.backbone == "sam2_t":
                ckpt_path = "/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/MemSAM_xgao083/exp_sam2_t_mem/Cityscapes/best_sam2_t.pth"

            ckpt = torch.load(ckpt_path, map_location="cpu")
            # Remove "module." prefix if present in the checkpoint keys.
            new_state_dict = {
                (k[len("module."):] if k.startswith("module.") else k): v
                for k, v in ckpt.items()
            }
            model.load_state_dict(new_state_dict)
    # Convert BatchNorm layers to SyncBatchNorm and wrap the model in DDP.
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch Single-GPU Training Example")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Input batch size for training (default: 1)")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=6e-5,
                        help="Learning rate (default: 6e-5)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed (default: 1234)")
    parser.add_argument("--save_root", default="",
                        help="Directory to save models")
    parser.add_argument("--exp_name", default="exp1",
                        help="Experiment name")
    parser.add_argument("--backbone", type=str, default="sam2_l",
                        help="Backbone model: sam2_l | sam2_b+ | sam2_s | sam2_t")
    parser.add_argument("--dataset", type=str, default="Stanford2D3D",
                        help="Dataset: Stanford2D3D | SynPASS | Cityscapes | Cityscapes19")
    parser.add_argument("--use_mem_bank", action="store_true",
                        help="Whether to use memory bank")
    parser.add_argument("--load_ckpt", action="store_true",
                        help="Whether to load checkpoints")

    args = parser.parse_args()

    # Initialize the distributed process group.
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main_process = (rank == 0)

    setup_seed(args.seed)
    save_path = os.path.join(args.save_root, args.exp_name, args.dataset)

    if is_main_process:
        print("File name:", __file__)
        os.makedirs(save_path, exist_ok=True)
        print(f"Use memory bank: {args.use_mem_bank}")

    writer = SummaryWriter(log_dir=save_path)

    train_loader, val_loader, test_loader, class_names, num_classes, num_maskmem = build_dataset_and_loader(args)
    args.num_classes = num_classes  # Allow build_sam_model to access the number of classes.

    if args.use_mem_bank:
        model = build_sam_model(args, num_maskmem, device, local_rank).to(device)
        if is_main_process:
            print(f"Using backbone: {args.backbone}, lr: {args.lr}, memory bank size: {num_maskmem}")
    else:
        model = build_sam_model(args, 0, device, local_rank).to(device)
        if is_main_process:
            print(f"Using backbone: {args.backbone}, memory bank size: 0")

    criterion_sup = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)

    total_iterations = args.num_epochs * len(train_loader)

    warmup_iters = 100 if args.load_ckpt else 1500

    best_performance = 0.0
    for epoch in range(args.num_epochs):
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion_sup,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=args.num_epochs,
            writer=writer,
            device=device,
            base_lr=args.lr,
            warmup_iters=warmup_iters,
            total_iterations=total_iterations,
            num_maskmem=num_maskmem,
            use_mem_bank=args.use_mem_bank,
            rank=rank,
            world_size=world_size,
            args=args,
        )

        best_performance = validation(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
            writer=writer,
            epoch=epoch,
            save_path=save_path,
            backbone_name=args.backbone,
            num_maskmem=num_maskmem,
            best_performance=best_performance,
            rank=rank,
            world_size=world_size,
            args=args,
        )

    writer.close()
    if is_main_process:
        print("Training Finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
