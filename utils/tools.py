import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import numpy as np
import torch

@torch.no_grad()
def merge_width_scatter_np(
    logits_np: np.ndarray,      # (N, B, C, H_patch, W_patch)
    orig_size: tuple,           # (H_full, W_full)
    W_patch: int,               # patch width
    stride_w: int,              # horizontal stride
    passes: int = 2,            # forward + backward = 2; only one pass = 1
    device: str = "cuda",
    return_probs: bool = False, # True to return (N,C,H,W) probabilities, otherwise return argmax
):
    """
    Fast merging along width direction using scatter_add_ to add all patch probabilities back to the original image.
    Assumptions:
      - Patches are extracted using sliding windows: from left to right for L patches; if passes=2, also from right to left
      - H_patch == H_full (otherwise, vertical offset handling is needed)
    """
    N, B, C, H_patch, Wp = logits_np.shape
    H_full, W_full = orig_size
    assert Wp == W_patch, "W_patch mismatch"
    assert H_patch == H_full, "Current implementation assumes no vertical splitting"

    # Number of patches per pass
    L = (W_full - W_patch) // stride_w + 1
    assert B % L == 0, f"B={B}, L={L}, passes={passes} mismatch"
    assert B // L == passes, f"passes={passes} inconsistent with input"

    # Convert numpy to torch
    logits = torch.from_numpy(logits_np).to(device=device, dtype=torch.float32)  # (N,B,C,H,Wp)

    # Softmax within each patch
    probs = torch.softmax(logits, dim=2)  # (N,B,C,H,Wp)

    # Reshape: (N,B,C,H,Wp) -> (N,passes,L,C,H,Wp)
    probs = probs.view(N, passes, L, C, H_patch, W_patch)

    # Align and average different passes (assuming pass=2 and the second is reverse)
    if passes == 1:
        probs = probs[:, 0]                      # (N,L,C,H,Wp)
    elif passes == 2:
        forward  = probs[:, 0]                   # (N,L,C,H,Wp)
        backward = probs[:, 1].flip(1)           # Reverse to match forward order
        probs = 0.5 * (forward + backward)       # (N,L,C,H,Wp)
    else:
        # General case: average all passes (adjust if special order is needed)
        probs = probs.mean(dim=1)                # (N,L,C,H,Wp)

    # Output tensor
    out = torch.zeros((N, C, H_full, W_full), device=device, dtype=probs.dtype)

    # Start column index for each patch
    starts = torch.arange(0, W_full - W_patch + 1, stride_w, device=device)  # (L,)
    # Column offsets within each patch
    offs = torch.arange(W_patch, device=device)  # (Wp,)
    # Global column indices for each patch
    idx = starts[:, None] + offs[None, :]        # (L, Wp)

    # Reorder probs to match scatter_add_ shape
    # probs: (N,L,C,H,Wp) -> (N,C,H,L,Wp)
    vals = probs.permute(0, 2, 3, 1, 4)          # (N,C,H,L,Wp)
    # Expand idx to match vals shape
    idx_exp = idx.view(1, 1, 1, L, W_patch).expand_as(vals)

    # Reshape to (N,C,H,L*Wp)
    vals_r = vals.reshape(N, C, H_full, L * W_patch)
    idx_r  = idx_exp.reshape(N, C, H_full, L * W_patch)

    # Perform scatter_add_ along width dimension
    out.scatter_add_(3, idx_r, vals_r)

    if return_probs:
        return out.cpu().numpy()  # (N,C,H,W)

    pred = out.argmax(dim=1).cpu().numpy()  # (N,H,W)
    return pred

@torch.no_grad()
def merge_width_scatter_np_uncertainty(
    logits_np: np.ndarray,      # (N, B, C, H_patch, W_patch)
    orig_size: tuple,           # (H_full, W_full)
    W_patch: int,               # patch width
    stride_w: int,              # horizontal stride
    passes: int = 2,            # forward + backward = 2; only one pass = 1
    device: str = "cuda",
    return_probs: bool = False, # True to return (N,C,H,W) probabilities, otherwise return argmax
    threshold = 0.5,
    ignore_index = 255,
):
    """
    Fast merging along width direction using scatter_add_ to add all patch probabilities back to the original image.
    Assumptions:
      - Patches are extracted using sliding windows: from left to right for L patches; if passes=2, also from right to left
      - H_patch == H_full (otherwise, vertical offset handling is needed)
    """
    N, B, C, H_patch, Wp = logits_np.shape
    H_full, W_full = orig_size
    assert Wp == W_patch, "W_patch mismatch"
    assert H_patch == H_full, "Current implementation assumes no vertical splitting"

    # Number of patches per pass
    L = (W_full - W_patch) // stride_w + 1
    assert B % L == 0, f"B={B}, L={L}, passes={passes} mismatch"
    assert B // L == passes, f"passes={passes} inconsistent with input"

    # Convert numpy to torch
    logits = torch.from_numpy(logits_np).to(device=device, dtype=torch.float32)  # (N,B,C,H,Wp)

    # Softmax within each patch
    probs = torch.softmax(logits, dim=2)  # (N,B,C,H,Wp)

    # Reshape: (N,B,C,H,Wp) -> (N,passes,L,C,H,Wp)
    probs = probs.view(N, passes, L, C, H_patch, W_patch)

    # Align and average different passes (assuming pass=2 and the second is reverse)
    if passes == 1:
        probs = probs[:, 0]                      # (N,L,C,H,Wp)
    elif passes == 2:
        forward  = probs[:, 0]                   # (N,L,C,H,Wp)
        backward = probs[:, 1].flip(1)           # Reverse to match forward order
        probs = 0.5 * (forward + backward)       # (N,L,C,H,Wp)
    else:
        # General case: average all passes (adjust if special order is needed)
        probs = probs.mean(dim=1)                # (N,L,C,H,Wp)

    # Output tensor
    out = torch.zeros((N, C, H_full, W_full), device=device, dtype=probs.dtype)

    # Start column index for each patch
    starts = torch.arange(0, W_full - W_patch + 1, stride_w, device=device)  # (L,)
    # Column offsets within each patch
    offs = torch.arange(W_patch, device=device)  # (Wp,)
    # Global column indices for each patch
    idx = starts[:, None] + offs[None, :]        # (L, Wp)

    # Reorder probs to match scatter_add_ shape
    # probs: (N,L,C,H,Wp) -> (N,C,H,L,Wp)
    vals = probs.permute(0, 2, 3, 1, 4)          # (N,C,H,L,Wp)
    # Expand idx to match vals shape
    idx_exp = idx.view(1, 1, 1, L, W_patch).expand_as(vals)

    # Reshape to (N,C,H,L*Wp)
    vals_r = vals.reshape(N, C, H_full, L * W_patch)
    idx_r  = idx_exp.reshape(N, C, H_full, L * W_patch)

    # Perform scatter_add_ along width dimension
    out.scatter_add_(3, idx_r, vals_r)

    if return_probs:
        return out.cpu().numpy()  # (N,C,H,W)

    probs_max, pred = out.max(dim=1)  # (N,H,W), (N,H,W)

    pred[probs_max < threshold] = ignore_index
    return pred.cpu().numpy()

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
