# Code with dataset loader for VOC12 and Cityscapes (adapted from bodokaiser/piwise code)
# Sept 2017
# Eduardo Romera
#######################
import torch
import random

import numpy as np
import os

from PIL import Image, ImageOps

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png") or filename.endswith("_100000.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}').replace('\\', '/')

def image_path_city(root, name):
    return os.path.join(root, f'{name}').replace('\\', '/')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class cityscapes(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, subset='val', target = True, crop_size=None, base_size=2048):
        self.images_root = os.path.join(root, 'leftImg8bit/' + subset).replace('\\', '/')
        self.labels_root = os.path.join(root, 'gtFine/' + subset).replace('\\', '/')
        self.target = target
        self.crop_size = crop_size
        self.base_size = base_size

        print (self.images_root)
        print (self.labels_root)
        self.filenames = [os.path.join(dp, f).replace('\\', '/') for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()
        # if (set == 'train'):
        #self.filenamesGt = [os.path.join(dp, f).replace('\\', '/') for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        # else:
        #     self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f).replace('\\', '/') for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
    
        self.filenamesGt.sort()
        print(len(self.filenames))
        print(len(self.filenamesGt))
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_mean =  [123.675, 116.28, 103.53]
        img_std = [58.395, 57.12, 57.375]

        #print(filename)

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        
        # if self.input_transform is not None:
        #     image = self.input_transform(image)

        #if (not self.target):
        if self.target:
            filenameGt = self.filenamesGt[index]
            with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
                label = load_image(f).convert('P')
            # print(np.array(label))
            # if self.target_transform is not None:
            #     label = self.target_transform(label)
            if self.crop_size is not None:
                image, label = self._sync_transform(image, label)
            if self.input_transform is not None:
                image = self.input_transform(image)
            if self.target_transform is not None:
                label = self.target_transform(label)
            
            # if filenameGt.endswith("_100000_labelTrainIds.png"):
            #     print(filenameGt)
            #     print(label)
            # print(type(image))
            
            # image = np.asarray(image).astype(np.float32)
            # image = (image - img_mean) / img_std
            # image = torch.Tensor(image)

            return image, label#, np.array(image.size()), filename
        
        return image, np.array(image.size()), filename

    def __len__(self):
        return len(self.filenames)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size # 480
        # random scale (short edge)
        w, h = img.size
        # range: (260 -> 1040); assume: (600, 400)
        long_size = random.randint(int(self.base_size*0.5), int(self.base_size*1.75))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            # left top Right Bottom
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # final transform, should be of size: (crop_size, crop_size)
        return img, mask#self._mask_transform(mask)
