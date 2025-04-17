import os.path as osp
import os.path as osp

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms

from .utils.transform import FixScaleRandomCropWH

class densepass13DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(2048, 400), sw_setting=None,
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val', ssl_dir='', trans='resize'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.ssl_dir = ssl_dir
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.sw_w_stride, self.sw_h_stride, self.sw_size = sw_setting
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        self.trans = trans
        self.img_source = ['Amsterdam', 'Athens', 'Auckland', 'Bangkok', 'Barcelona', 'Bremen', 'Brussel', 
                            'Buenosaires', 'Canberra', 'Capetown', 'Chicago', 'Copenhagen', 'Dublin', 'Edinburgh',
                            'Florence', 'Glasgow', 'Helsinki', 'Hochiminhcity', 'Istanbul', 'Jakarta', 'Lisbon', 
                            'Manila', 'Marseille', 'Melbourne', 'Mexicocity', 'Montreal', 'Moscow', 'Nottingham', 
                            'Osaka', 'Oslo', 'Sandiego', 'Saopaulo', 'Seoul', 'Singapore', 'Stockholm', 'Toronto',
                            'Turin', 'Yokohama', 'Zagreb', 'Zurich']
        # self.target_source = ['Turin', 'Yokohama', 'Zagreb', 'Zurich']
        # self.target_source = ['Osaka', 'Oslo', 'Sandiego', 'Saopaulo', 'Seoul', 'Singapore', 'Stockholm', 'Toronto']
        # self.target_source = ['Manila', 'Marseille', 'Melbourne', 'Mexicocity', 'Montreal', 'Moscow', 'Nottingham']
        # self.target_source = ['Florence', 'Glasgow', 'Helsinki', 'Hochiminhcity', 'Istanbul', 'Jakarta', 'Lisbon']
        # self.target_source = ['Buenosaires', 'Canberra', 'Capetown', 'Chicago', 'Copenhagen', 'Dublin', 'Edinburgh']
        # self.target_source = ['Amsterdam', 'Athens', 'Auckland', 'Bangkok', 'Barcelona', 'Bremen', 'Brussel']
        self.target_source = ['Amsterdam', 'Athens', 'Auckland', 'Bangkok', 'Barcelona', 'Bremen', 'Brussel', 
                            'Buenosaires', 'Canberra', 'Capetown', 'Chicago', 'Copenhagen', 'Dublin', 'Edinburgh',
                            'Florence', 'Glasgow', 'Helsinki', 'Hochiminhcity', 'Istanbul', 'Jakarta', 'Lisbon', 
                            'Manila', 'Marseille', 'Melbourne', 'Mexicocity', 'Montreal', 'Moscow', 'Nottingham', 
                            'Osaka', 'Oslo', 'Sandiego', 'Saopaulo', 'Seoul', 'Singapore', 'Stockholm', 'Toronto',
                            'Turin', 'Yokohama', 'Zagreb', 'Zurich']

        for target_name in self.target_source:
            for name in self.img_ids:
                if target_name in name:
                    img_file = osp.join(self.root, "leftImg8bit/%s" % (name))
                    self.files.append({
                        "img": img_file,
                        "name": name
                    })

    def __len__(self):
        return len(self.files)

    def _sliding_windows(self, image, window_size, h_stride, w_stride):
        _, h, w = image.shape
        patches_image = []
        for y in range(0, h, h_stride):
            for x in range(0, w, w_stride):
                if x + window_size <= w and y+ window_size <= h:
                    patches_image.append(image[:, y:y + window_size, x:x + window_size])
        patches_image.extend(reversed(patches_image))
        return torch.stack(patches_image)

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
        name = datafiles["name"]

        image = image.resize(self.crop_size, Image.BICUBIC)


        size = np.asarray(image, np.float32).shape

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])
        image = input_transform(image)
        image = self._sliding_windows(image, window_size=self.sw_size, h_stride=self.sw_h_stride, w_stride=self.sw_w_stride)

        return image, size, name

if __name__ == '__main__':
    dst = densepass13TestDataSet("data/DensePASS_train_pseudo_val", 'dataset/densepass_list/val.txt', mean=(0,0,0))
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
