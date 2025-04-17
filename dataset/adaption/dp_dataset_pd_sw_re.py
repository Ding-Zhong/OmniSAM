import os.path as osp
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as F
import pdb

class densepass19TestDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 400), sw_setting=None, mean=(128, 128, 128),
                 scale=False, mirror=False, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.sw_w_stride, self.sw_h_stride, self.sw_size = sw_setting
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbname = name.replace(".png", "labelTrainIds.png")
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, lbname))
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
    
    def _coords_list(self):
        w, h = self.crop_size
        coords_list = []
        for y in range(0, h, self.sw_h_stride):
            for x in range(0, w, self.sw_w_stride):
                if x + self.sw_size <= w and y+ self.sw_size <= h:
                    coords_list.append((y, x))
        
        coords_list.extend(reversed(coords_list))
        
        return coords_list

    
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

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        # # 3) 为了得到 2048×1024，需要对高度再进行补足：1024 - 400 = 624
        # target_width, target_height = 2048, 1024
        # w, h = image.size
        # pad_h = target_height - h
        # pad_w = target_width - w 

        # if pad_w < 0 or pad_h < 0:
        #     raise ValueError("Pad width/height cannot be negative, please check your crop_size settings.")

        # image = F.pad(image, (0, 0, pad_w, pad_h), fill=(0, 0, 0))
        # label = F.pad(label, (0, 0, pad_w, pad_h), fill=self.ignore_label)
        
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))
        ori_label = torch.LongTensor(np.array(ori_label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.sw_size, h_stride=self.sw_h_stride, w_stride=self.sw_w_stride)
        size = np.asarray(label).shape 

        return image, label, size, name, ori_label

class densepass19ValDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 400), sw_setting=None, mean=(128, 128, 128),
                 scale=False, mirror=False, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.set = set
        self.sw_w_stride, self.sw_h_stride, self.sw_size = sw_setting
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            lbname = name.replace(".png", "labelTrainIds.png")
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, lbname))
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
    
    def _sliding_windows(self, image, label, window_size, h_stride, w_stride):
        h, w = label.shape
        patches_image = []
        patches_label = []
        for y in range(0, h, h_stride):
            for x in range(0, w, w_stride):
                if x + window_size <= w and y+ window_size <= h:
                    patches_image.append(image[:, y:y + window_size, x:x + window_size])
                    patches_label.append(label[y:y + window_size, x:x + window_size])

        return torch.stack(patches_image), torch.stack(patches_label)
    
    def _coords_list(self):
        w, h = self.crop_size
        coords_list = []
        for y in range(0, h, self.sw_h_stride):
            for x in range(0, w, self.sw_w_stride):
                if x + self.sw_size <= w and y+ self.sw_size <= h:
                    coords_list.append((y, x))
        
        return coords_list

    
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

        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        # # 3) 为了得到 2048×1024，需要对高度再进行补足：1024 - 400 = 624
        # target_width, target_height = 2048, 1024
        # w, h = image.size
        # pad_h = target_height - h
        # pad_w = target_width - w 

        # if pad_w < 0 or pad_h < 0:
        #     raise ValueError("Pad width/height cannot be negative, please check your crop_size settings.")

        # image = F.pad(image, (0, 0, pad_w, pad_h), fill=(0, 0, 0))
        # label = F.pad(label, (0, 0, pad_w, pad_h), fill=self.ignore_label)
        
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        label = torch.LongTensor(np.array(label).astype('int32'))
        image, label = self._sliding_windows(image, label, window_size=self.sw_size, h_stride=self.sw_h_stride, w_stride=self.sw_w_stride)
        size = np.asarray(label).shape 

        return image, label, size, name

if __name__ == '__main__':
    root_syn = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/data/DensePASS'
    val_list = '/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dataset/adaption/densepass_list/val.txt'
    sw_w_stride, sw_h_stride, sw_size = 768, 512, 1024  
    val_dataset = densepass13TestDataSet(root_syn, val_list, crop_size=(4096, 1024), sw_setting=(sw_w_stride, sw_h_stride, sw_size))
    trainloader = data.DataLoader(val_dataset, batch_size=4)
    pdb.set_trace()
    for i, data in enumerate(trainloader):
        imgs, labels, *args = data
        img1 = imgs[1]
        if i == 0:
            img = torchvision.utils.make_grid(img1).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img))
            img.save("/hpc2hdd/JH_DATA/share/xzheng287/PrivateShareGroup/xzheng287_360/DATR-main/dp13_samples.jpg")
        break
