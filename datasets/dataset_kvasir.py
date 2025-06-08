import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import train_test_split


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        img_size=224,
        transform=None,
        split="train",
        test_size=0.2,
        random_state=42,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.transform = transform
        self.split = split

        # Get all image and mask filenames
        all_images = sorted(os.listdir(images_dir))
        all_masks = sorted(os.listdir(masks_dir))

        assert len(all_images) == len(all_masks), "Images and masks count mismatch!"

        # Split the data into train and test sets
        train_images, test_images, train_masks, test_masks = train_test_split(
            all_images, all_masks, test_size=test_size, random_state=random_state
        )

        # Select appropriate split
        if split == "train":
            self.images = train_images
            self.masks = train_masks
        elif split == "test":
            self.images = test_images
            self.masks = test_masks
        else:
            raise ValueError("split must be either 'train' or 'test'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

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
