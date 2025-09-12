import os
import random
from PIL import Image

import torch
import numpy as np
from torch.utils import data
from torchvision import transforms

class DensePASS13(data.Dataset):
    def __init__(self, root, list_path, crop_size=(2048, 400), sliding_window=None, passes=1, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set = set
        self.stride, self.window_size = sliding_window
        self.files = []
        assert passes in [1, 2]
        self.passes = passes
        
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        for name in self.img_ids:
            img_file = os.path.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbname = name.replace(".png", "labelTrainIds.png")
            label_file = os.path.join(self.root, "gtFine/%s/%s" % (self.set, lbname))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name[1:]
            })
        self._key = np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12])

    def __len__(self):
        return len(self.files)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        for value in values:
            if value == 255: 
                new_mask[mask == value] = 255
            else:
                new_mask[mask == value] = self._key[value]
        return new_mask
    
    def _sliding_windows(self, image, label, window_size, stride):
        h, w = label.shape
        patched_image = []
        patched_label = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x:x + window_size])
                patched_label.append(label[:, x:x + window_size])
        
        if self.passes == 2:
            patched_image.extend(reversed(patched_image))
            patched_label.extend(reversed(patched_label))

        return torch.stack(patched_image), torch.stack(patched_label)

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt
    
    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        label = self._map19to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        ori_label = label
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))
        ori_label = torch.LongTensor(np.array(ori_label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)
        size = np.asarray(label).shape 

        return image, label, size, name, ori_label
