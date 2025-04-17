import os.path as osp
import os.path as osp

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms

import random

class FixScaleRandomCropWH(object):
    def __init__(self, crop_size_wh, is_label=False):
        assert isinstance(crop_size_wh, tuple)
        self.crop_size = crop_size_wh
        self.is_label = is_label

    def __call__(self, sample):
        w, h = sample.size
        cw, ch = self.crop_size
        if w < cw:
            # new w h
            nw = cw
            nh = int(1.0 * h * nw / w)
            sample = sample.resize((nw, nh), Image.NEAREST if self.is_label else Image.BILINEAR)
        w, h = sample.size
        if h < ch:
            # new w h
            nh = ch
            nw = int(1.0 * nh * w / h)
            sample = sample.resize((nw, nh), Image.NEAREST if self.is_label else Image.BILINEAR)

        # # center crop
        # w, h = sample.size
        # x1 = int(round((w - self.crop_size[0]) / 2.))
        # y1 = int(round((h - self.crop_size[1]) / 2.))
        # random crop crop_size
        w, h = sample.size
        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        # left, upper, right, and lower
        sample = sample.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        return sample

class densepassDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 400),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', ssl_dir='', trans='resize'):
        self.root = root
        self.label_root = '/hy-tmp/workplace/xuzheng/CVPR/images'
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.ssl_dir = ssl_dir
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.trans = trans
        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s" % (name))
            lbname = name.replace(".jpg", "_labelTrainIds.png")
            label_file = osp.join(self.label_root, "pesudo_label_final/%s" % (lbname))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        if self.trans == 'resize':
            # resize
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)

        size = np.asarray(image, np.float32).shape

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))
  
        return image, label,  np.array(size), name
