import torch
import os
import numpy as np
from torch.utils.data import dataloader
from torchvision import transforms
import random
from PIL import Image, ImageOps


root_voc2012 = '/data/ssd/datasets/cityscapes/citys_cps/'

class ValCityDataset(torch.utils.data.Dataset):
    CLASSES_NAME = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign',
                'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70],
                [102, 102, 156], [190, 153, 153], [153, 153, 153],
                [250, 170, 30], [220, 220, 0], [107, 142, 35],
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
                [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                [0, 0, 230], [119, 11, 32]]
              
    def __init__(self, root_dir, split='train_aug',use_difficult=False, is_train = True, augment = None, base_size=2048, crop_size=769, file_length=None, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5],not_cps=True,my_slide=False):
        self.root_voc2012= root_voc2012 # /hpc/users/CONNECT/yunhaoluo/datasets/citys
        self.txt_path = root_dir
        self.use_difficult=use_difficult
        self.imgset=split
        #self.cutmix = cutmix
        self.num_class = 19
        self._imgpath = os.path.join(self.root_voc2012, "", "%s")
        self._labelpath = os.path.join(self.root_voc2012, "", "%s")

        self._imgsetpath = os.path.join(self.txt_path, "%s.txt")
        self._labelsetpath = os.path.join(self.txt_path, "%s.txt")

        with open(self._imgsetpath%self.imgset, 'r') as f:
            self.img_label_pairs=f.readlines()
        
        self.img_label_ids = [x.strip().split('\t') for x in self.img_label_pairs]
        
        self.img_ids = [x[0].strip() for x in self.img_label_ids]
        self.label_ids = [x[1].strip() for x in self.img_label_ids]

        self.name2id=dict(zip(ValCityDataset.CLASSES_NAME,range(len(ValCityDataset.CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}
        # self.mean=[0.485, 0.456, 0.406]
        # self.std=[0.229, 0.224, 0.225]
        self.train = is_train
        self.augment = augment
        self.base_size = base_size
        self.crop_size = crop_size
        self.norm_mean= norm_mean
        self.norm_std = norm_std
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            # make img to [-1, 1]
            transforms.Normalize(self.norm_mean, self.norm_std),
        ])
        self.target_transform = None
        self.not_cps = not_cps
        self.my_slide = my_slide

        print("INFO=====> CityScapes dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,index):

        img_id=self.img_ids[index]
        label_id=self.label_ids[index]

        _img = Image.open(self._imgpath%img_id).convert('RGB')
        _target = Image.open(self._labelpath%label_id)
        # print(f'img: {img_id}; label: {label_id}')
        # synchrosized transform, img is pil, target is tensor now 
        if self.train:
            _img, _target = self._sync_transform( _img, _target)
        elif self.my_slide:
            _target = torch.from_numpy(np.array(_target)).long()
        else:
            _img, _target = self._val_sync_transform( _img, _target)
            # _target = torch.from_numpy(np.array(_target)).long()
            pass

        # general resize, normalize and toTensor
        if self.my_slide or (self.img_transform is not None and self.not_cps):
            "image need to be transfromed to tensor here"
            _img = self.img_transform(_img)
        # is None for now
        if self.target_transform is not None:
            _target = self.target_transform(_target)

        return _img, _target
    
    def my_collate(self,batch):
        '''
        ** Resize images and retain original ratio **
        image = [item[0] for item in batch]
        class_label = [item[1][:,0] for item in batch]
        '''
        return dataloader.default_collate(batch)

       
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size # 480
        # random scale (short edge)
        w, h = img.size
        # range: (260 -> 1040); assume: (600, 400)
        long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
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
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size # 480,480
        short_size = outsize
        w, h = img.size
        # make short_size to crop_size: 480 for resize
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        # w, h = img.size
        # x1 = int(round((w - outsize) / 2.))
        # y1 = int(round((h - outsize) / 2.))
        # img = img.crop((x1, y1, x1+outsize, y1+outsize))
        # mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _construct_new_file_names(self, length): # length is the required length
        # 183 -> 91xx
        assert isinstance(length, int)
        print(f'len {length}, img {len(self.img_ids)}')
        files_len = len(self.img_ids) # 原来一轮迭代的长度

        # 仅使用小部分数据
        if length < files_len:
            return self.img_ids[:length]
        
        # 按照设定获得的一轮迭代的长度
        new_file_names = self.img_ids * (length // files_len)

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self.img_ids[i] for i in new_indices]
        
        self.img_ids = new_file_names
        print(f'{self.txt_path}: {len(self.img_ids)}')
        print(self.img_ids[0], self.img_ids[183])

        return None
        
    def get_length(self):
        return len(self.img_ids)
