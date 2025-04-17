import numpy as np
import torch
import torch.nn as nn

def sliding_window_prediction(
    logits_maps: np.ndarray,
    coords_list: np.ndarray,
    orig_size: tuple
):

    N, B, C, H, W = logits_maps.shape
    orig_H, orig_W = orig_size
    
    vote_map = np.zeros((N, orig_H, orig_W, C, 2), dtype=np.float32)

    for n in range(N):
        for b in range(B):
            top_x, top_y = coords_list[n, b]
            
            sub_logits = logits_maps[n, b]

            sub_pred_class = np.argmax(sub_logits, axis=0)  # (H, W)
            sub_max_logit  = np.max(sub_logits, axis=0)     # (H, W)
            
            x_indices = np.arange(H) + top_x  # [top_x, top_x+1, ..., top_x+H-1]
            y_indices = np.arange(W) + top_y  # [top_y, top_y+1, ..., top_y+W-1]

            valid_x = (x_indices >= 0) & (x_indices < orig_H)
            valid_y = (y_indices >= 0) & (y_indices < orig_W)
            
            if not np.any(valid_x) or not np.any(valid_y):
                continue  
            
            x_indices = x_indices[valid_x]
            y_indices = y_indices[valid_y]
            
            xx, yy = np.meshgrid(x_indices, y_indices, indexing='ij')
            
            sub_x = np.arange(H)[valid_x]  # e.g. [0, 1, 2, ...] 
            sub_y = np.arange(W)[valid_y]
            sub_xx, sub_yy = np.meshgrid(sub_x, sub_y, indexing='ij')
            

            # (M,)；M = len(x_indices)*len(y_indices)
            sub_pred_class_2d = sub_pred_class[sub_xx, sub_yy]   # (len(x_indices), len(y_indices))
            sub_max_logit_2d  = sub_max_logit[sub_xx, sub_yy]
            classes_flat = sub_pred_class_2d.ravel()
            logits_flat  = sub_max_logit_2d.ravel()
            
            x_flat = xx.ravel()
            y_flat = yy.ravel()
            
            np.add.at(vote_map, (n, x_flat, y_flat, classes_flat, 0), 1)
            
            current_logits = vote_map[n, x_flat, y_flat, classes_flat, 1]
            mask = logits_flat > current_logits
            
            if np.any(mask):
                x_update = x_flat[mask]
                y_update = y_flat[mask]
                c_update = classes_flat[mask]
                new_vals = logits_flat[mask]
                
                vote_map[n, x_update, y_update, c_update, 1] = new_vals
    
    counts = vote_map[..., 0]  # (N, orig_H, orig_W, C)
    confs  = vote_map[..., 1]  # (N, orig_H, orig_W, C)

    max_count = np.max(counts, axis=-1, keepdims=True)
    
    candidate_mask = (counts == max_count)  # (N, orig_H, orig_W, C)
    
    confs_candidate = np.where(candidate_mask, confs, -np.inf)
    best_class = np.argmax(confs_candidate, axis=-1)  # (N, orig_H, orig_W)
    
    return best_class


def sliding_window_prediction_uncertainty(
    logits_maps: np.ndarray,
    coords_list: np.ndarray,
    orig_size: tuple,
    threshold: float
):
    N, B, C, H, W = logits_maps.shape
    orig_H, orig_W = orig_size

    coverage_map = np.zeros((N, orig_H, orig_W), dtype=np.int32)
    vote_map = np.zeros((N, orig_H, orig_W, C, 3), dtype=np.float32)
    vote_map[..., 2] = np.inf  

    for n in range(N):
        for b in range(B):
            top_x, top_y = coords_list[n, b]
            sub_logits = logits_maps[n, b]
            
            sub_pred_class = np.argmax(sub_logits, axis=0)
            sub_max_logit  = np.max(sub_logits, axis=0)
            
            x_indices = np.arange(H) + top_x
            y_indices = np.arange(W) + top_y

            valid_x = (x_indices >= 0) & (x_indices < orig_H)
            valid_y = (y_indices >= 0) & (y_indices < orig_W)
            if not np.any(valid_x) or not np.any(valid_y):
                continue

            x_indices = x_indices[valid_x]
            y_indices = y_indices[valid_y]

            xx, yy = np.meshgrid(x_indices, y_indices, indexing='ij')

            sub_x = np.arange(H)[valid_x]
            sub_y = np.arange(W)[valid_y]
            sub_xx, sub_yy = np.meshgrid(sub_x, sub_y, indexing='ij')

            sub_pred_class_2d = sub_pred_class[sub_xx, sub_yy]   # shape (len_x, len_y)
            sub_max_logit_2d  = sub_max_logit[sub_xx, sub_yy]

            classes_flat = sub_pred_class_2d.ravel()   # shape (M,)
            logits_flat  = sub_max_logit_2d.ravel()    # shape (M,)
            x_flat       = xx.ravel()                  # shape (M,)
            y_flat       = yy.ravel()                  # shape (M,)

            coverage_map[n, x_flat, y_flat] += 1
            np.add.at(vote_map, (n, x_flat, y_flat, classes_flat, 0), 1)

            current_max_logits = vote_map[n, x_flat, y_flat, classes_flat, 1]
            mask_max = logits_flat > current_max_logits
            if np.any(mask_max):
                x_update = x_flat[mask_max]
                y_update = y_flat[mask_max]
                c_update = classes_flat[mask_max]
                new_vals = logits_flat[mask_max]
                vote_map[n, x_update, y_update, c_update, 1] = new_vals

            current_min_logits = vote_map[n, x_flat, y_flat, classes_flat, 2]
            mask_min = logits_flat < current_min_logits
            if np.any(mask_min):
                x_update = x_flat[mask_min]
                y_update = y_flat[mask_min]
                c_update = classes_flat[mask_min]
                new_vals = logits_flat[mask_min]
                vote_map[n, x_update, y_update, c_update, 2] = new_vals

    best_class = np.full((N, orig_H, orig_W), 255, dtype=np.uint8)
    class_counts = vote_map[..., 0]  # (N, H, W, C)
    min_logits   = vote_map[..., 2]  # (N, H, W, C)

    cond1 = (class_counts == coverage_map[..., None])  # shape (N, H, W, C)
    cond2 = (min_logits > threshold)                   # shape (N, H, W, C)
    candidate_mask = cond1 & cond2                     # shape (N, H, W, C)

    sum_candidates = candidate_mask.sum(axis=-1)       # shape (N, H, W)
    argmax_candidates = candidate_mask.argmax(axis=-1) # shape (N, H, W)

    covered_mask = (coverage_map > 0)
    unique_mask = (sum_candidates == 1) & covered_mask

    best_class[unique_mask] = argmax_candidates[unique_mask]

    return best_class

def create_overlay(mask, color_map: dict, alpha: int = 127):
    """
    Create an overlay image from mask and color mapping.
    
    Returns a NumPy array of overlay data.
    """
    h, w = mask.shape
    overlay_data = np.zeros((h, w, 4), dtype=np.uint8)
    for label, color in color_map.items():
        indices = (mask == label)
        overlay_data[indices, 0] = color[0]  # R
        overlay_data[indices, 1] = color[1]  # G
        overlay_data[indices, 2] = color[2]  # B
        overlay_data[indices, 3] = alpha     # A
    return overlay_data

def load_model_weights(model: nn.Module, ckpt_path: str, remove_module_prefix: bool = False):
    """
    Load model weights from checkpoint.
    If remove_module_prefix is True, remove the "module." prefix from keys.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if remove_module_prefix:
        new_state_dict = {k[len("module."):]: v if k.startswith("module.") else v for k, v in ckpt.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(ckpt)