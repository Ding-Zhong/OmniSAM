import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from torchvision import transforms
import pdb

class CS19SrcDataSet_sw(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), sw_setting=None, mean=(128, 128, 128), 
                scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.sw_w_stride, self.sw_h_stride, self.sw_size = sw_setting
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            name = os.path.basename(name)
            name = name.replace("_leftImg8bit", "")
            img_file = osp.join(self.root, "images/%s/%s" % (self.set, name))
            lbname = name.replace(".png", "_gtFine_labelTrainIds.png")
            label_file = osp.join(self.root, "segmentation/%s/%s" % (self.set, lbname))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12])

    def __len__(self):
        return len(self.files)

    def _coords_list(self):
        w, h = self.crop_size
        coords_list = []
        for y in range(0, h, self.sw_h_stride):
            for x in range(0, w, self.sw_w_stride):
                if x + self.sw_size <= w and y+ self.sw_size <= h:
                    coords_list.append((y, x))
        
        return coords_list

    def _sliding_windows(self, image, label, window_size, h_stride, w_stride):
        h, w = label.shape
        patches_image = []
        patches_label = []
        for y in range(0, h, h_stride):
            for x in range(0, w, w_stride):
                if x + window_size <= w and y+ window_size <= h:
                    patches_image.append(image[:, y:y + window_size, x:x + window_size])
                    patches_label.append(label[y:y + window_size, x:x + window_size])
        if np.random.randint(0, 2):
            patches_image.reverse()
            patches_label.reverse()
        return torch.stack(patches_image), torch.stack(patches_label)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        # new_mask -= 1
        for value in values:
            if value == 255: 
                new_mask[mask==value] = 255
            else:
                new_mask[mask==value] = self._key[value]
        mask = new_mask
        return mask

    def _get_num_maskmem(self):
        w, h = self.crop_size
        window_size = self.sw_size
        h_stride, w_stride = self.sw_h_stride, self.sw_w_stride
        sub_num_h = (h - window_size) // h_stride + 1
        sub_num_w = (w - window_size) // w_stride + 1
        num_maskmem = sub_num_h * sub_num_w
        return num_maskmem

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = np.array(label).astype('int32')
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
        image, label = self._sliding_windows(image, label, window_size=self.sw_size, h_stride=self.sw_h_stride, w_stride=self.sw_w_stride)
        return image, label, np.array(size), name


class CS19SrcDataSetval_sw(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), sw_setting=None, mean=(128, 128, 128), 
                scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.set = set
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.sw_w_stride, self.sw_h_stride, self.sw_size = sw_setting
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            name = os.path.basename(name)
            name = name.replace("_leftImg8bit", "")
            img_file = osp.join(self.root, "images/%s/%s" % (self.set, name))
            lbname = name.replace(".png", "_gtFine_labelTrainIds.png")
            label_file = osp.join(self.root, "segmentation/%s/%s" % (self.set, lbname))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self._key = np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12])

    def __len__(self):
        return len(self.files)

    def _coords_list(self):
        w, h = self.crop_size
        coords_list = []
        for y in range(0, h, self.sw_h_stride):
            for x in range(0, w, self.sw_w_stride):
                if x + self.sw_size <= w and y+ self.sw_size <= h:
                    coords_list.append((y, x))

        coords_list.extend(reversed(coords_list))
        
        return coords_list
        
    def _sliding_windows(self, image, label, window_size, h_stride, w_stride):
        h, w = label.shape
        patches_image = []
        patches_label = []
        for y in range(0, h, h_stride):
            for x in range(0, w, w_stride):
                if x + window_size <= w and y+ window_size <= h:
                    patches_image.append(image[:, y:y + window_size, x:x + window_size])
                    patches_label.append(label[y:y + window_size, x:x + window_size])

        patches_image.extend(reversed(patches_image))
        patches_label.extend(reversed(patches_label))
        
        return torch.stack(patches_image), torch.stack(patches_label)

    def _map19to13(self, mask):
        values = np.unique(mask)
        new_mask = np.ones_like(mask) * 255
        # new_mask -= 1
        for value in values:
            if value == 255: 
                new_mask[mask==value] = 255
            else:
                new_mask[mask==value] = self._key[value]
        mask = new_mask
        return mask

    def _get_num_maskmem(self):
        w, h = self.crop_size
        window_size = self.sw_size
        h_stride, w_stride = self.sw_h_stride, self.sw_w_stride
        sub_num_h = (h - window_size) // h_stride + 1
        sub_num_w = (w - window_size) // w_stride + 1
        num_maskmem = sub_num_h * sub_num_w
        return num_maskmem

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        label = np.array(label).astype('int32')
        label = Image.fromarray(label)
        ori_label = label
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
        ori_label = torch.LongTensor(np.array(ori_label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.sw_size, h_stride=self.sw_h_stride, w_stride=self.sw_w_stride)
        return image, label, np.array(size), name, ori_label

if __name__ == '__main__':
    dst = CSSrcDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
