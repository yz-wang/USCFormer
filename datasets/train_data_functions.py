import os
import re

import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()
        gt_dir = os.path.join(train_data_dir, 'leftImg8bit')
        foggy_dir = os.path.join(train_data_dir, 'leftImg8bit_foggy')

        self.crop_size = crop_size
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
        crop_width, crop_height = self.crop_size
        n = len(self.gt_images)
        gt = Image.open(self.gt_images[index % n]).convert('RGB')
        foggy = Image.open(self.foggy_images[index % n]).convert('RGB')

        width, height = foggy.size

        if width < crop_width and height < crop_height :
            foggy = foggy.resize((crop_width,crop_height), Image.ANTIALIAS)
            gt = gt.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width :
            foggy = foggy.resize((crop_width,height), Image.ANTIALIAS)
            gt = gt.resize((crop_width,height), Image.ANTIALIAS)
        elif height < crop_height :
            foggy = foggy.resize((width,crop_height), Image.ANTIALIAS)
            gt = gt.resize((width, crop_height), Image.ANTIALIAS)

        width, height = foggy.size

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        foggy_crop_img = foggy.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt.crop((x, y, x + crop_width, y + crop_height))

        # --- Transform to tensor --- #
        transform_foggy = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        foggy = transform_foggy(foggy_crop_img)
        gt = transform_gt(gt_crop_img)

        return foggy, gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.foggy_images)




