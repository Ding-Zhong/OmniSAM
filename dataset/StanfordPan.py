import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch.utils import data
import glob

__FOLD__ = {
    '1_train': ['area_1', 'area_2', 'area_3', 'area_4', 'area_6'],
    '1_val': ['area_5a', 'area_5b'],
    '2_train': ['area_1', 'area_3', 'area_5a', 'area_5b', 'area_6'],
    '2_val': ['area_2', 'area_4'],
    '3_train': ['area_2', 'area_4', 'area_5a', 'area_5b'],
    '3_val': ['area_1', 'area_3', 'area_6'],
    'trainval': ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6'],
}

def _get_stanford2d3d_path(folder, fold, mode='train'):
    '''image is jpg, label is png'''
    img_paths = []
    if mode == 'train':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    elif mode == 'val':
        area_ids = __FOLD__['{}_{}'.format(fold, mode)]
    elif mode == 'trainval':
        area_ids = __FOLD__[mode]
    else:
        raise NotImplementedError
    for a in area_ids:
        img_paths += glob.glob(os.path.join(folder, '{}/pano/rgb/*_rgb.png'.format(a)))
    img_paths = sorted(img_paths)
    return img_paths

class StanfordPan8forPL(data.Dataset):
    def __init__(self, root, list_path, crop_size=(2048, 1024), sliding_window=None, set='val', fold=1):
        self.root = root
        self.crop_size = crop_size
        self.img_paths = _get_stanford2d3d_path(root, fold, set)
        self.stride, self.window_size = sliding_window

        self.files = []
        # --- stanford color2id
        with open('dataset/s2d3d_pin_list/semantic_labels.json') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open('dataset/s2d3d_pin_list/name2label.json') as f:
            name2id = json.load(f)
        self.colors = np.load('dataset/s2d3d_pin_list/colors.npy') # for visualization
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)

        for p in self.img_paths:
            self.files.append({
                "img": p,
                "label": p.replace("rgb", "semantic"),
                "name": p.split(self.root+'/')[-1]
            })
        self._key = np.array([255,255,255,0,1,255,255,2,3,4,5,6,7])

    def __len__(self):
        return len(self.files)

    def _map13to8(self, mask):
        values = np.unique(mask)
        for value in values:
            if value == 255: 
                mask[mask==value] = 255
            else:
                mask[mask==value] = self._key[value]
        return mask

    def _sliding_windows(self, image, window_size, stride):
        _, h, w = image.shape
        patched_image = []
        for x in range(0, w, stride):
            if x + window_size <= w:
                patched_image.append(image[:, :, x:x + window_size])
        patched_image.extend(reversed(patched_image))
        return torch.stack(patched_image)

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt

    def _color2id(self, img, sem):
        sem = np.array(sem, np.int32)
        rgb = np.array(img, np.int32)
        unk = (sem[..., 0] != 0)
        sem = self.id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0
        sem[rgb.sum(-1) == 0] = 0
        sem -= 1 # 0->255
        return Image.fromarray(sem)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        width, height = image.size
        left, top, right, bottom = 0, 320, width, 1728
        image = image.crop((left, top, right, bottom)) 
        name = datafiles["name"]
        image = image.resize(self.crop_size, Image.BICUBIC)

        size = np.asarray(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        image = self._sliding_windows(image, window_size=self.window_size, stride=self.stride)

        return image, np.array(size), name

class StanfordPan8forAdaptation(data.Dataset):
    def __init__(self, root, list_path, crop_size=(2048, 1024), sliding_window=None,  set='val', fold=1):
        self.root = root
        self.crop_size = crop_size
        self.img_paths = _get_stanford2d3d_path(root, fold, set)
        self.stride, self.window_size = sliding_window

        self.files = []
        # --- stanford color2id
        with open('dataset/s2d3d_pin_list/semantic_labels.json') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open('dataset/s2d3d_pin_list/name2label.json') as f:
            name2id = json.load(f)
        self.colors = np.load('dataset/s2d3d_pin_list/colors.npy') # for visualization
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)

        for p in self.img_paths:
            self.files.append({
                "img": p,
                "label": p.replace("rgb", "semantic"),
                "name": p.split(self.root+'/')[-1]
            })
        self._key = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    def __len__(self):
        return len(self.files)

    def _map13to8(self, mask):
        values = np.unique(mask)
        for value in values:
            if value == 255: 
                mask[mask==value] = 255
            else:
                mask[mask==value] = self._key[value]
        return mask

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

    def _get_frame_cnt(self):
        w, h = self.crop_size
        frame_cnt = (w - self.window_size) // self.stride + 1
        return frame_cnt

    def _color2id(self, img, sem):
        sem = np.array(sem, np.int32)
        rgb = np.array(img, np.int32)
        unk = (sem[..., 0] != 0)
        sem = self.id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0
        sem[rgb.sum(-1) == 0] = 0
        sem -= 1 # 0->255
        return Image.fromarray(sem)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        label = self._map13to8(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        size = np.asarray(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        label = torch.LongTensor(np.array(label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)

        return image, label, np.array(size), name


class StanfordPan8forVal(data.Dataset):
    def __init__(self, root, list_path, crop_size=(2048, 1024), sliding_window=None, passes=1, set='val', fold=1):
        self.root = root
        self.crop_size = crop_size
        self.img_paths = _get_stanford2d3d_path(root, fold, set)
        self.stride, self.window_size = sliding_window

        self.files = []
        with open('dataset/s2d3d_pin_list/semantic_labels.json') as f:
            id2name = [name.split('_')[0] for name in json.load(f)] + ['<UNK>']
        with open('dataset/s2d3d_pin_list/name2label.json') as f:
            name2id = json.load(f)
        self.colors = np.load('dataset/s2d3d_pin_list/colors.npy') # for visualization
        self.id2label = np.array([name2id[name] for name in id2name], np.uint8)

        for p in self.img_paths:
            self.files.append({
                "img": p,
                "label": p.replace("rgb", "semantic"),
                "name": p.split(self.root+'/')[-1]
            })
        self._key = np.array([255,255,255,0,1,255,255,2,3,4,5,6,7])

    def __len__(self):
        return len(self.files)

    def _map13to8(self, mask):
        values = np.unique(mask)
        for value in values:
            if value == 255: 
                mask[mask==value] = 255
            else:
                mask[mask==value] = self._key[value]
        return mask

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

    def _color2id(self, img, sem):
        sem = np.array(sem, np.int32)
        rgb = np.array(img, np.int32)
        unk = (sem[..., 0] != 0)
        sem = self.id2label[sem[..., 1] * 256 + sem[..., 2]]
        sem[unk] = 0
        sem[rgb.sum(-1) == 0] = 0
        sem -= 1 # 0->255
        return Image.fromarray(sem)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        width, height = image.size
        left, top, right, bottom = 0, 320, width, 1728
        image = image.crop((left, top, right, bottom))        
        label = label.crop((left, top, right, bottom))

        label = self._color2id(image, label)
        label = self._map13to8(np.array(label).astype('int32'))
        label = Image.fromarray(label)
        name = datafiles["name"]


        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        ori_label = label
        size = np.asarray(image).shape
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)

        label = torch.LongTensor(np.array(label).astype('int32'))
        ori_label = torch.LongTensor(np.array(ori_label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.window_size, stride=self.stride)

        return image, label, np.array(size), name, ori_label