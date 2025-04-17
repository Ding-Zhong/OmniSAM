import os.path as osp
import os.path as osp

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms

from .utils.transform import FixScaleRandomCropWH


class synpass13DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(1024, 1024), mean=(128, 128, 128), 
                scale=True, mirror=True, ignore_label=255, set='val', trans = 'FixScaleRandomCropWH'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.trans = trans
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        for name in self.img_ids:
            img_file = osp.join(self.root, name)
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

    def _map23to13(self, mask):
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

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        # 映射标签
        label = self._map23to13(np.array(label).astype('int32'))
        label = Image.fromarray(label)

        name = datafiles["name"]

        # 准备一个转换，把 PIL Image -> Tensor，并做标准化
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

        if self.trans == 'resize':
            # 如果只是固定缩放
            image = image.resize(self.crop_size, Image.BICUBIC)
            label = label.resize(self.crop_size, Image.NEAREST)

            # 转成 Tensor
            image = input_transform(image)
            label = torch.LongTensor(np.array(label).astype('int32'))

            # 计算尺寸（H, W, C）
            size = np.array(image.permute(1,2,0)).shape  # 注意这里要反一下维度

            return image, label, size, name

        elif self.trans == 'FixScaleRandomCropWH':
            # ============= 关键修改点：多次随机裁剪  =============
            n_crops = 5  # 比如一次裁剪 5 张

            image_crops = []
            label_crops = []
            sizes = []  # 如果你还想记录每次裁剪后的尺寸

            for _ in range(n_crops):
                img_cropped = FixScaleRandomCropWH(self.crop_size, is_label=False)(image)
                lbl_cropped = FixScaleRandomCropWH(self.crop_size, is_label=True)(label)

                # 转成 Tensor
                img_cropped_tensor = input_transform(img_cropped)
                lbl_cropped_tensor = torch.LongTensor(np.array(lbl_cropped).astype('int32'))

                # 保存
                image_crops.append(img_cropped_tensor)
                label_crops.append(lbl_cropped_tensor)

                # 如果还需要尺寸信息
                h, w = np.array(img_cropped).shape[:2]  # (H, W)
                sizes.append((h, w))

            # ----------------------------------------------------
            # 这里返回 “多张裁剪结果” 的方式可以有很多：
            # 1) 返回 list：由外部使用时自己处理
            # 2) 直接堆叠成一个 Tensor，形状 [n_crops, C, H, W]
            # 3) 两者兼顾
            # ----------------------------------------------------

            # 下面演示把图片堆叠起来，标签也堆叠起来：
            # 如果不想堆叠，可以直接返回两个 list
            image_stack = torch.stack(image_crops, dim=0)  # [n_crops, 3, H, W]
            label_stack = torch.stack(label_crops, dim=0)  # [n_crops, H, W]

            # 返回堆叠好的 Tensor
            # size 和 name 可以只返回一次，也可以返回多份，看你需求
            return image_stack, label_stack, sizes, name

        else:
            raise NotImplementedError



if __name__ == '__main__':
    dst = synpass13DataSet("data/SynPASS", 'dataset/SynPASS_list/val.txt', mean=(0,0,0))
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels, *args = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            img = Image.fromarray(np.uint8(img) )
            img.show()
        break
