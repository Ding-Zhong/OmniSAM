import os
from PIL import Image
import numpy as np

from utils.io import ensure_dir

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
    if dataset_name == "Stanford2D3D":
        root_file = "data/Stanford2d3d_Seg"
        target_size = (3072, 1024)
        for idx, img_name in enumerate(name_list):
            img_path = os.path.join(root_file, img_name)
            image_save_path = os.path.join(save_path, img_name)
            ensure_dir(os.path.dirname(image_save_path))

            ori_img = Image.open(img_path).convert('RGBA')
            ori_img = ori_img.crop((0, 320, ori_img.width, 1728))
            ori_img.save(image_save_path)

            label_save_path = image_save_path.replace("rgb", "semantic")
            overlay_save_path = image_save_path.replace("rgb", "overlay")

            pseudo_label = Image.fromarray(mask_list[idx].astype(np.uint8), mode='L')
            pseudo_label = pseudo_label.resize(target_size, Image.NEAREST)
            ensure_dir(os.path.dirname(label_save_path))
            pseudo_label.save(label_save_path)
            print(f"Pseudo label saved to: {label_save_path}")

            overlay_img = create_overlay(mask_list[idx], target_size)
            ori_resized = ori_img.resize(target_size, Image.BICUBIC)
            out_img = Image.alpha_composite(ori_resized, overlay_img)
            ensure_dir(os.path.dirname(overlay_save_path))
            out_img.save(overlay_save_path)
            print(f"Overlay image saved to: {overlay_save_path}")
    else:
        root_file = "data/WildPASS"
        save_root_overlap = os.path.join(save_path, "gtFine_overlap")
        save_root_label = os.path.join(save_path, "gtFine")
        ensure_dir(save_root_overlap)
        ensure_dir(save_root_label)
        target_size = (2048, 400)
        for idx, img_name in enumerate(name_list):
            img_path = os.path.join(root_file, "leftImg8bit", img_name)
            save_img_path = os.path.join(save_root_overlap, img_name).replace(".jpg", ".png")
            ori_img = Image.open(img_path).convert('RGBA')

            label_name = img_name.replace(".jpg", "labelTrainIds.png")
            label_save_path = os.path.join(save_root_label, label_name)
            pseudo_label = Image.fromarray(mask_list[idx].astype(np.uint8), mode='L')
            ensure_dir(os.path.dirname(label_save_path))
            pseudo_label.save(label_save_path)
            # print(f"Pseudo label saved to: {label_save_path}")

            overlay_img = create_overlay(mask_list[idx], target_size)
            out_img = Image.alpha_composite(ori_img, overlay_img)
            ensure_dir(os.path.dirname(save_img_path))
            out_img.save(save_img_path)
            # print(f"Overlay image saved to: {save_img_path}")