import os
import sys
import cv2
from math import pi
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import torchvision
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
def pair(t):
    return t if isinstance(t, tuple) else (t, t)    

def eq2tangent(img, height=224, width=224):   
    #img = img.astype(np.float32) / 255
    [erp_h, erp_w, _] = img.shape
    # print(erp_h, erp_w)
    # plt.imshow(img[:,:,[2,1,0]])
    # plt.show()

    # img_new = img.astype(np.float32) / 255
    # img_new = np.transpose(img_new, [2, 0, 1])
    # img_new = torch.from_numpy(img_new)
    # img_new = img_new.unsqueeze(0)
    img_new = torch.tensor(img, dtype=torch.float32) / 255
    img_new = img_new.permute(2,0,1).unsqueeze(0)
    FOV = [120, 120]
    FOV = [FOV[0]/360.0, FOV[1]/180.0]
    FOV = torch.tensor(FOV, dtype=torch.float32)
    PI = math.pi
    PI_2 = math.pi * 0.5
    PI2 = math.pi * 2
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
    screen_points = torch.stack([xx.flatten(), yy.flatten()], -1)
    # print(screen_points.shape)
    # num_rows = 6
    # num_cols = [3, 8, 12, 12, 8, 3]
    # phi_centers = [-75, -45, -15, 15, 45, 75]
    num_rows = 4
    num_cols = [3, 6, 6, 3]
    phi_centers = [-67.5, -22.5, 22.5, 67.5]
    # num_rows = 3
    # num_cols = [3, 4, 3]
    # phi_centers = [-60, 0, 60]
    phi_interval = 180 // num_rows
    all_combos = []
    erp_mask = []
    for i, n_cols in enumerate(num_cols):
        for j in np.arange(n_cols):
            theta_interval = 360 / n_cols
            theta_center = j * theta_interval + theta_interval / 2
            center = [theta_center, phi_centers[i]]
            all_combos.append(center)
            up = phi_centers[i] + phi_interval / 2
            down = phi_centers[i] - phi_interval / 2
            left = theta_center - theta_interval / 2
            right = theta_center + theta_interval / 2
            up = int((up + 90) / 180 * erp_h)
            down = int((down + 90) / 180 * erp_h)
            left = int(left / 360 * erp_w)
            right = int(right / 360 * erp_w)
            mask = np.zeros((erp_h, erp_w), dtype=int)
            mask[down:up, left:right] = 1
            erp_mask.append(mask)
    all_combos = np.vstack(all_combos) 
    shifts = np.arange(all_combos.shape[0]) * width
    shifts = torch.from_numpy(shifts).float()
    erp_mask = np.stack(erp_mask)
    erp_mask = torch.from_numpy(erp_mask).float()
    n_patch = all_combos.shape[0]
    # print(all_combos)
    center_point = torch.from_numpy(all_combos).float()  # -180 to 180, -90 to 90
    center_point[:, 0] = (center_point[:, 0]) / 360  #0 to 1
    center_point[:, 1] = (center_point[:, 1] + 90) / 180  #0 to 1
    cp = center_point * 2 - 1
    cp[:, 0] = cp[:, 0] * PI
    cp[:, 1] = cp[:, 1] * PI_2
    cp = cp.unsqueeze(1)
    convertedCoord = screen_points * 2 - 1
    convertedCoord[:, 0] = convertedCoord[:, 0] * PI
    convertedCoord[:, 1] = convertedCoord[:, 1] * PI_2
    convertedCoord = convertedCoord * (torch.ones(screen_points.shape, dtype=torch.float32) * FOV)
    convertedCoord = convertedCoord.unsqueeze(0).repeat(cp.shape[0], 1, 1)
    # print(cp.shape)
    # print(convertedCoord.shape)
    x = convertedCoord[:, :, 0]
    y = convertedCoord[:, :, 1]

    rou = torch.sqrt(x ** 2 + y ** 2)
    c = torch.atan(rou)
    sin_c = torch.sin(c)
    cos_c = torch.cos(c)
    lat = torch.asin(cos_c * torch.sin(cp[:, :, 1]) + (y * sin_c * torch.cos(cp[:, :, 1])) / rou)
    lon = cp[:, :, 0] + torch.atan2(x * sin_c, rou * torch.cos(cp[:, :, 1]) * cos_c - y * torch.sin(cp[:, :, 1]) * sin_c)
    lat_new = lat / PI_2 
    lon_new = lon / PI 
    lon_new[lon_new > 1] -= 2
    lon_new[lon_new<-1] += 2 
    # print(lon_new.shape, lat_new.shape)
    lon_new = lon_new.view(1, n_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, n_patch*width)
    lat_new = lat_new.view(1, n_patch, height, width).permute(0, 2, 1, 3).contiguous().view(height, n_patch*width)
    grid = torch.stack([lon_new, lat_new], -1)
    grid = grid.unsqueeze(0).to('cuda')
    persp = F.grid_sample(img_new, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    # print(persp.shape)
    persp_int = persp[0].permute(1, 2, 0)#.numpy()
    persp_int = persp_int * 255
    # cv2.imwrite('pers.png', persp_int.astype(np.uint8))
    # print(persp_int.shape)
    # plt.figure(figsize=(20, 12))
    # plt.imshow(persp_int[:,:,[2,1,0]].astype(np.uint8), aspect=1)
    # plt.show()
    return persp_int