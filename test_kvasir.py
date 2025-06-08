import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from datasets.dataset_kvasir import KvasirSegDataset
from networks.DAEFormer import DAEFormer as DAEFormer_orig
from networks.DAEFormer_new import DAEFormer as DAEFormer_new
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import torch.nn as nn


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
    parser.add_argument(
        "--model_version",
        type=str,
        default="new",
        choices=["original", "new"],
        help="DAEFormer model version to use (default: new)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    num_workers = 4

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
        images_dir,
        masks_dir,
        img_size=args.img_size,
        transform=transform,
        split="test",
        test_size=0.2,
        random_state=42,  # Use same seed as training
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model selection
    if args.model_version == "original":
        model = DAEFormer_orig(num_classes=1).to(device)
        print("Using original DAEFormer implementation")
    else:
        model = DAEFormer_new(num_classes=1).to(device)
        print("Using new DAEFormer implementation (WaveletAttention)")

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
    test_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs.squeeze(1), masks.squeeze(1))
            test_loss += loss.item() * images.size(0)

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

    # Calculate average test loss
    test_loss = test_loss / len(test_dataset)

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)

    # Calculate metrics
    f1 = f1_score(all_gts, all_preds)
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    std_dice = np.std(dice_scores)
    std_iou = np.std(iou_scores)

    # Print comprehensive results
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Mean IoU Score: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"F1 Score (pixel): {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_gts, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Background", "Foreground"]
    )
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Pixel-wise Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\nConfusion matrix saved as confusion_matrix.png")

    # Save test results to file
    results = {
        "test_loss": test_loss,
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "f1_score": f1,
    }

    with open("test_results.txt", "w") as f:
        f.write("Test Results:\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}\n")
        f.write(f"Mean IoU Score: {mean_iou:.4f} ± {std_iou:.4f}\n")
        f.write(f"F1 Score (pixel): {f1:.4f}\n")

    print("\nDetailed results saved to test_results.txt")


if __name__ == "__main__":
    main()
