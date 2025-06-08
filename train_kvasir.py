import argparse
import logging
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as T
from networks.DAEFormer import DAEFormer as DAEFormer_orig
from networks.DAEFormer_new import DAEFormer as DAEFormer_new
from datasets.dataset_kvasir import KvasirSegDataset
from tqdm import tqdm


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
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed (default: 1234)"
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="Whether to use deterministic training (default: 1)",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="new",
        choices=["original", "new"],
        help="DAEFormer model version to use (default: new)",
    )

    return parser.parse_args()


def setup_logging(args):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info(str(args))


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def plot_loss_graph(epoch_losses, epoch_avg_losses, val_losses, save_path):
    """Plot and save the epoch vs loss graph with both current and average losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(epoch_losses) + 1),
        epoch_losses,
        "b-",
        label="Train Loss",
        alpha=0.7,
    )
    plt.plot(
        range(1, len(epoch_avg_losses) + 1),
        epoch_avg_losses,
        "r-",
        label="Train Avg Loss",
        alpha=0.7,
    )
    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        "g-",
        label="Validation Loss",
        alpha=0.7,
    )
    plt.title("Training and Validation Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def main():
    args = parse_args()

    # Setup logging
    setup_logging(args)

    # Set random seed
    set_seed(args.seed, args.deterministic)

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Dataset and DataLoader
    transform = T.Compose(
        [
            T.Resize((args.img_size, args.img_size)),
            T.ToTensor(),
        ]
    )

    # Create train dataset
    train_dataset = KvasirSegDataset(
        args.images_dir,
        args.masks_dir,
        img_size=args.img_size,
        transform=transform,
        split="train",
        test_size=0.2,
        random_state=args.seed,
    )

    # Create validation dataset (using 20% of training data)
    val_dataset = KvasirSegDataset(
        args.images_dir,
        args.masks_dir,
        img_size=args.img_size,
        transform=transform,
        split="test",
        test_size=0.2,
        random_state=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model, Loss, Optimizer
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    # Select model version
    if args.model_version == "original":
        model = DAEFormer_orig(num_classes=args.num_classes).to(device)
        logging.info("Using original DAEFormer implementation")
    else:
        model = DAEFormer_new(num_classes=args.num_classes).to(device)
        logging.info("Using new DAEFormer implementation")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    logging.info(f"Starting training on {device}")
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Number of epochs: {args.num_epochs}")

    best_loss = float("inf")
    epoch_losses = []  # List to store epoch losses for plotting
    epoch_avg_losses = []  # List to store epoch average losses for plotting
    val_losses = []  # List to store validation losses for plotting

    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        # Progress bar for training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")

        for batch_idx, (images, masks) in enumerate(pbar):
            batch_start_time = time.time()

            # Move data to device
            images, masks = images.to(device, non_blocking=True), masks.to(
                device, non_blocking=True
            )
            data_load_time = time.time() - batch_start_time

            optimizer.zero_grad()

            # Forward pass
            forward_start_time = time.time()
            outputs = model(images)
            forward_time = time.time() - forward_start_time

            outputs = outputs.squeeze(1)  # [B, H, W]
            loss = criterion(outputs, masks.squeeze(1))

            # Backward pass
            backward_start_time = time.time()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start_time

            running_loss += loss.item() * images.size(0)

            # Update progress bar with timing information
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{running_loss/((batch_idx+1)*args.batch_size):.4f}",
                    "data_time": f"{data_load_time:.3f}s",
                    "forward_time": f"{forward_time:.3f}s",
                    "backward_time": f"{backward_time:.3f}s",
                }
            )

        epoch_loss = running_loss / len(train_dataset)
        epoch_avg_loss = running_loss / ((batch_idx + 1) * args.batch_size)
        epoch_losses.append(epoch_loss)
        epoch_avg_losses.append(epoch_avg_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
            for images, masks in val_pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = outputs.squeeze(1)
                loss = criterion(outputs, masks.squeeze(1))
                val_loss += loss.item() * images.size(0)
                val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        val_loss = val_loss / len(val_dataset)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start_time

        # Log epoch statistics
        logging.info(
            f"Epoch [{epoch+1}/{args.num_epochs}] "
            f"Train Loss: {epoch_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Time: {epoch_time:.2f}s"
        )

        # Plot and save loss graph after each epoch
        plot_loss_graph(
            epoch_losses,
            epoch_avg_losses,
            val_losses,
            os.path.join(args.save_dir, f"loss_graph_{args.model_version}.png"),
        )

        # Save best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(
                args.save_dir, f"best_model_{args.model_version}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "model_version": args.model_version,
                },
                best_model_path,
            )
            logging.info(f"New best model saved: {best_model_path}")

        # Save regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.save_dir, f"daeformer_{args.model_version}_epoch{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "model_version": args.model_version,
                },
                checkpoint_path,
            )
            logging.info(f"Checkpoint saved: {checkpoint_path}")

    logging.info("Training complete!")
    logging.info(f"Best validation loss achieved: {best_loss:.4f}")


if __name__ == "__main__":
    main()
