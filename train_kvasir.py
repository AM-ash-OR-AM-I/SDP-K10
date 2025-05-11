import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from networks.DAEFormer_new import DAEFormer
from datasets.dataset_kvasir import KvasirSegDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DAEFormer on Kvasir-SEG dataset"
    )

    # Dataset paths
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Path to Kvasir-SEG images directory",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        required=True,
        help="Path to Kvasir-SEG masks directory",
    )

    # Training hyperparameters
    parser.add_argument(
        "--img_size", type=int, default=224, help="Input image size (default: 224)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of output classes (default: 1 for binary segmentation)",
    )

    # Model and training options
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for data loading (default: 2)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Save model every N epochs (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training (default: cuda)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create save directory if it doesn't exist
    import os

    os.makedirs(args.save_dir, exist_ok=True)

    # Dataset and DataLoader
    transform = T.Compose(
        [
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
        ]
    )

    dataset = KvasirSegDataset(
        args.images_dir, args.masks_dir, img_size=args.img_size, transform=transform
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # Model, Loss, Optimizer
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    model = DAEFormer(num_classes=args.num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)  # [B, H, W]
            loss = criterion(outputs, masks.squeeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {epoch_loss:.4f}")

        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.save_dir, f"daeformer_kvasir_epoch{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()
