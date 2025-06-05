import numpy as np
from PIL import Image

import os

npz_dir = "datasets/Synapse/train_npz"
output_img_dir = "datasets/Synapse/train_npz/images"
output_mask_dir = "datasets/Synapse/train_npz/labels"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)


for npz_path in os.listdir(npz_dir):
    if not npz_path.startswith("case0005_slice"):
        continue
    data = np.load(os.path.join(npz_dir, npz_path), allow_pickle=True)
    img = data["image"]
    label = data["label"]
    # If needed, transpose or squeeze to get (H, W, C) for images and (H, W) for masks
    if img.shape[0] in [1, 3] and img.ndim == 3:  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))
    if label.ndim == 3:
        label = label.squeeze()
    Image.fromarray((img * 255).astype("uint8")).save(
        f"{output_img_dir}/{npz_path.replace('.npz', '.png')}"
    )
    Image.fromarray((label * 255).astype("uint8")).save(
        f"{output_mask_dir}/{npz_path.replace('.npz', '.png')}"
    )
