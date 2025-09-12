import os
from PIL import Image

import torch
import numpy as np
from torch.utils import data
from torchvision import transforms

class SynPASS13(data.Dataset):
    def __init__(self, root, list_path, crop_size=(2048, 1024), sliding_window=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.stride, self.window_size = sliding_window
        self.files = []

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        for name in self.img_ids:
            img_file = os.path.join(self.root, name)
            lbname = name.replace("img", "semantic").replace('.jpg', '_trainID.png')
            label_file = osp.join(self.root, lbname)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([255,2,4,255,11,5,0,0,1,8,12,3,7,10,255,255,255,255,6,255,255,255,9])

    def __len__(self):
        return len(self.files)
    
    def _sliding_windows(self, image, label, window_size, stride):
        h, w = label.shape
        patched_image = []
        patched_label = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x:x + window_size])
                patched_label.append(label[:, x:x + window_size])
        if np.random.randint(0, 2):
            patched_image.reverse()
            patched_label.reverse()
        return torch.stack(patched_image), torch.stack(patched_label)
    
    def _map23to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        for value in values:
            if value == 255: 
                new_mask[mask==value] = 255
            else:
                new_mask[mask==value] = self._key[value]
        mask = new_mask
        return mask

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt
    
    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map23to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        size = np.array(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)
        return image, label, np.array(size), name

class SynPASS13forVal(data.Dataset):
    def __init__(self, root, list_path, crop_size=(2048, 1024), sliding_window=None):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.stride, self.window_size = sliding_window
        self.files = []

        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        for name in self.img_ids:
            img_file = os.path.join(self.root, name)
            lbname = name.replace("img", "semantic").replace('.jpg', '_trainID.png')
            label_file = osp.join(self.root, lbname)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([255,2,4,255,11,5,0,0,1,8,12,3,7,10,255,255,255,255,6,255,255,255,9])

    def __len__(self):
        return len(self.files)
    
    def _sliding_windows(self, image, label, window_size, stride):
        h, w = label.shape
        patched_image = []
        patched_label = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x:x + window_size])
                patched_label.append(label[:, x:x + window_size])
        if np.random.randint(0, 2):
            patched_image.reverse()
            patched_label.reverse()
        return torch.stack(patched_image), torch.stack(patched_label)
    
    def _map23to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        for value in values:
            if value == 255: 
                new_mask[mask==value] = 255
            else:
                new_mask[mask==value] = self._key[value]
        mask = new_mask
        return mask

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt
    
    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = self._map23to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        size = np.array(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)
        return image, label, np.array(size), name
