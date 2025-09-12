import os
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

from utils.io import ensure_dir
from utils.vis import mask_vis 
from utils.tools import merge_width_scatter_np, merge_width_scatter_np_uncertainty


class PseudoLabelGenerator:
    def __init__(self, model, device, class_names, num_classes):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = num_classes

    def run(self, dataloader, save_dir: str, max_num: int, dataset_name: str, uc_threshold: float):
        self.model.eval()
        used_paths = []
        # total_steps = math.ceil(max_num / B)
        total_steps = max_num

        progress_bar = tqdm(enumerate(dataloader), total=total_steps, desc="Pseudolabel Generation")
        with torch.no_grad():
            for i, batch in progress_bar:
                images, _, names = batch
                used_paths.extend(names)

                images = images.to(self.device)
                B = images.shape[0]
                frame_cnt = images.shape[1] // 2
                H, W = images.shape[-2:]

                logits_maps = np.zeros((B, 2 * frame_cnt, len(self.class_names), H, W))

                for frame_idx in range(frame_cnt):
                    frame_img = images[:, frame_idx]
                    output, _, _ = self.model(frame_idx if dataset_name != 'Stanford2D3D' else 0, frame_img)
                    logits_maps[:, frame_idx] = output.cpu().numpy()

                self.model.memory_bank._init_output_dict_source()

                for frame_idx in range(frame_cnt):
                    frame_img = images[:, frame_cnt + frame_idx]
                    output, _, _ = self.model(frame_idx if dataset_name != 'Stanford2D3D' else 0, frame_img)
                    logits_maps[:, frame_cnt + frame_idx] = output.cpu().numpy()

                self.model.memory_bank._init_output_dict_source()
                # start = time.perf_counter()
                merged_pred = merge_width_scatter_np_uncertainty(
                    logits_np=logits_maps,
                    orig_size=(1024, 4096) if dataset_name != 'Stanford2D3D' else (1024, 4096),
                    W_patch=W,
                    stride_w=384,
                    passes=2,
                    device='cuda',
                    return_probs=False,
                    threshold = uc_threshold
                )
                mask_vis(merged_pred, names, dataset_name, save_dir, None)  # args 可传 None 或精简版
                if i + 1 >= total_steps:
                    break
        return used_paths