import os

import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        gt_dir = os.path.join(val_data_dir, 'leftImg8bit')
        foggy_dir = os.path.join(val_data_dir, 'leftImg8bit_foggy')

        # image path
        self.gt_images = []
        self.foggy_images = []


        for file_name in sorted(os.listdir(foggy_dir)):
            self.foggy_images.append(
                    os.path.join(foggy_dir, file_name))  # ./data/cityscapes/train/leftImg8bit_foggy/0001.png
        for file_name in sorted(os.listdir(gt_dir)):      
            for i in range(3):
                self.gt_images.append(
                    os.path.join(gt_dir, file_name))  # ./data/cityscapes/train/leftImg8bit/0001.png


    def get_images(self, index):
        n = len(self.gt_images)
        gt = Image.open(self.gt_images[index % n]).convert('RGB')
        foggy = Image.open(self.foggy_images[index % n]).convert('RGB')

        foggy_path = self.foggy_images[index % n]

        # Resizing image in the multiple of 16"
        wd_new, ht_new = foggy.size

        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        wd_new = int(224 * np.ceil(wd_new / 224.0))
        ht_new = int(224 * np.ceil(ht_new / 224.0))
        foggy = foggy.resize((wd_new, ht_new), Image.ANTIALIAS)
        gt = gt.resize((wd_new, ht_new), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_foggy = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        foggy = transform_foggy(foggy)
        gt = transform_gt(gt)

        return foggy, gt, foggy_path

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.foggy_images)
