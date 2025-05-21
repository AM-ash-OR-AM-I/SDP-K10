import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from datasets.dataset_kvasir import KvasirSegDataset
from networks.DAEFormer import DAEFormer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Test DAEFormer on Kvasir-SEG dataset")
    parser.add_argument(
        "--volume_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Input image size (default: 224)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for testing (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for testing (default: cuda)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    # Data paths
    images_dir = os.path.join("Kvasir-SEG", "images")
    masks_dir = os.path.join("Kvasir-SEG", "masks")

    # Dataset and DataLoader
    transform = T.Compose(
        [
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
        ]
    )
    test_dataset = KvasirSegDataset(
        images_dir, masks_dir, img_size=args.img_size, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = DAEFormer(num_classes=1).to(device)
    checkpoint = torch.load(args.volume_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Inference
    dice_scores = []
    iou_scores = []
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()
            # Compute Dice and IoU
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = (preds + masks).sum(dim=(1, 2, 3))
            dice = (2.0 * intersection) / (union + 1e-8)
            iou = intersection / (union - intersection + 1e-8)
            dice_scores.extend(dice.cpu().numpy())
            iou_scores.extend(iou.cpu().numpy())
            # For confusion matrix and F1
            all_preds.append(preds.cpu().numpy().astype(np.uint8).flatten())
            all_gts.append(masks.cpu().numpy().astype(np.uint8).flatten())

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)

    # F1 Score (same as Dice for binary)
    f1 = f1_score(all_gts, all_preds)
    print(f"Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"F1 Score (pixel): {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_gts, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Background", "Foreground"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Pixel-wise Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    main()
