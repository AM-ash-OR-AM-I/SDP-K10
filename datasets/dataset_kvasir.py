import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

class KvasirSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=224, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

        assert len(self.images) == len(self.masks), "Images and masks count mismatch!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Resize and ToTensor
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = T.ToTensor()(T.Resize((self.img_size, self.img_size))(image))
            mask = T.ToTensor()(T.Resize((self.img_size, self.img_size))(mask))

        # Normalize mask to 0/1
        mask = (mask > 0.5).float()
        return image, mask