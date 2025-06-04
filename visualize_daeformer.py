from cv2 import transform
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from networks.DAEFormer import DAEFormer
import os


def visualize_tensor(tensor, title, save_path=None):
    """Visualize a tensor as an image"""
    # Convert tensor to numpy array
    if tensor.dim() == 4:  # [B, C, H, W]
        tensor = tensor[0]  # Take first image in batch

    # If tensor has multiple channels, take mean across channels
    if tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0)

    # Ensure tensor is 2D for visualization
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    # Normalize to [0, 1]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    plt.figure(figsize=(8, 8))
    plt.imshow(tensor.detach().cpu().numpy(), cmap="viridis")
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_attention_maps(
    model, image_tensor, save_dir="all_visualizations/daeformer"
):
    """Visualize the complete transformation process of DAEFormer"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Initial Image
    plt.figure(figsize=(8, 8))
    if image_tensor.size(1) == 1:
        plt.imshow(image_tensor[0, 0].detach().cpu().numpy(), cmap="gray")
    else:
        plt.imshow(image_tensor[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.title("1. Input Image")
    plt.axis("off")
    plt.savefig(f"{save_dir}/1_input.png")
    plt.close()

    with torch.no_grad():
        # Get encoder outputs
        if image_tensor.size(1) == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        # Get encoder outputs
        encoder_outputs = model.backbone(image_tensor)

        # Visualize each encoder stage with more details
        for i, output in enumerate(encoder_outputs):
            # Get the shape information
            b, c, h, w = output.shape
            plt.figure(figsize=(8, 8))

            # Take mean across channels for visualization
            vis_output = output[0].mean(dim=0)
            # Normalize for visualization
            vis_output = (vis_output - vis_output.min()) / (
                vis_output.max() - vis_output.min() + 1e-8
            )

            plt.imshow(vis_output.detach().cpu().numpy(), cmap="viridis")
            plt.title(f"2. Encoder Stage {i+1}\nShape: {c} channels, {h}x{w} spatial")
            plt.colorbar(label="Activation Strength")
            plt.axis("off")
            plt.savefig(f"{save_dir}/2_encoder_stage_{i+1}.png")
            plt.close()

        # 3. Decoder Stages
        b, c, _, _ = encoder_outputs[2].shape

        # Decoder Stage 1
        tmp_2 = model.decoder_2(encoder_outputs[2].permute(0, 2, 3, 1).view(b, -1, c))
        decoder_1_output = tmp_2.view(b, -1, 28, 28).permute(0, 3, 1, 2)
        plt.figure(figsize=(8, 8))
        vis_output = decoder_1_output[0].mean(dim=0)
        vis_output = (vis_output - vis_output.min()) / (
            vis_output.max() - vis_output.min() + 1e-8
        )
        plt.imshow(vis_output.detach().cpu().numpy(), cmap="viridis")
        plt.title("3. Decoder Stage 1\nUpsampling from 14x14 to 28x28")
        plt.colorbar(label="Activation Strength")
        plt.axis("off")
        plt.savefig(f"{save_dir}/3_decoder_stage_1.png")
        plt.close()

        # Decoder Stage 2
        tmp_1 = model.decoder_1(tmp_2, encoder_outputs[1].permute(0, 2, 3, 1))
        decoder_2_output = tmp_1.view(b, -1, 56, 56).permute(0, 3, 1, 2)
        plt.figure(figsize=(8, 8))
        vis_output = decoder_2_output[0].mean(dim=0)
        vis_output = (vis_output - vis_output.min()) / (
            vis_output.max() - vis_output.min() + 1e-8
        )
        plt.imshow(vis_output.detach().cpu().numpy(), cmap="viridis")
        plt.title("3. Decoder Stage 2\nUpsampling from 28x28 to 56x56")
        plt.colorbar(label="Activation Strength")
        plt.axis("off")
        plt.savefig(f"{save_dir}/3_decoder_stage_2.png")
        plt.close()

        # Decoder Stage 3 (Final Output)
        tmp_0 = model.decoder_0(tmp_1, encoder_outputs[0].permute(0, 2, 3, 1))
        plt.figure(figsize=(8, 8))
        # For final output, we want to see the actual segmentation
        final_output = torch.sigmoid(
            tmp_0[0, 0]
        )  # Take first channel and apply sigmoid
        plt.imshow(final_output.detach().cpu().numpy(), cmap="gray")
        plt.title("4. Final Output\nSegmentation Mask (224x224)")
        plt.colorbar(label="Probability")
        plt.axis("off")
        plt.savefig(f"{save_dir}/4_final_output.png")
        plt.close()

        print(f"Transformation visualizations saved to {save_dir}")
        print("Visualization sequence:")
        print("1. Input Image")
        print("2. Encoder Stages (3 stages, progressively smaller spatial dimensions)")
        print("3. Decoder Stages (2 stages, progressively larger spatial dimensions)")
        print("4. Final Output (Segmentation mask)")


def visualize_segmentation_with_gt(
    image_path, mask_path, model, save_dir="segmentation_results"
):
    """Visualize segmentation results with ground truth comparison for a single image"""
    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess image and mask
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Load image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Convert to tensors
    image_tensor = transform(image).unsqueeze(0)
    mask_tensor = transform(mask).unsqueeze(0)

    # Ensure mask is binary (0 or 1)
    mask_bin = (mask_tensor > 0.5).float()

    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        # For multi-class model, take the first channel as binary prediction
        if output.size(1) > 1:
            output = output[:, 0:1]  # Take first channel
        output = torch.sigmoid(output)
        pred_bin = (output > 0.5).float()

    # Only consider the foreground (mask == 1 or pred == 1)
    mask_fg = mask_bin[0, 0]
    pred_fg = pred_bin[0, 0]

    intersection = (pred_fg * mask_fg).sum()
    union = pred_fg.sum() + mask_fg.sum() - intersection
    dice = (2.0 * intersection) / (pred_fg.sum() + mask_fg.sum() + 1e-8)
    iou = intersection / (union + 1e-8)

    # Debug prints
    print("Mask sum (foreground pixels):", mask_fg.sum().item())
    print("Prediction sum (foreground pixels):", pred_fg.sum().item())
    print("Intersection:", intersection.item())
    print("Union:", union.item())

    # Create visualization
    plt.figure(figsize=(15, 6))

    # Input image
    plt.subplot(131)
    plt.imshow(image_tensor[0].permute(1, 2, 0).cpu().numpy())
    plt.title("Input Image")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(132)
    plt.imshow(mask_fg.cpu().numpy(), cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Model prediction
    plt.subplot(133)
    plt.imshow(pred_fg.cpu().numpy(), cmap="gray")
    plt.title(f"Model Prediction\nDice: {dice.item():.3f}, IoU: {iou.item():.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/segmentation_comparison.png")
    plt.close()

    # Create overlay visualization
    plt.figure(figsize=(10, 10))

    # Convert tensors to numpy arrays
    pred_np = pred_fg.cpu().numpy()
    gt_np = mask_fg.cpu().numpy()

    # Create RGB visualization
    overlay = np.zeros((224, 224, 3))
    overlay[..., 0] = gt_np  # Red channel for ground truth
    overlay[..., 1] = pred_np  # Green channel for prediction
    overlay[..., 2] = 0  # Blue channel empty

    plt.imshow(image_tensor[0].permute(1, 2, 0).cpu().numpy())
    plt.imshow(overlay, alpha=0.5)
    plt.title("Overlay: Red=Ground Truth, Green=Prediction\nYellow=Overlap")
    plt.axis("off")
    plt.savefig(f"{save_dir}/segmentation_overlay.png")
    plt.close()

    print(f"Segmentation visualizations saved to {save_dir}")
    print(f"Dice Score: {dice.item():.4f}")
    print(f"IoU Score: {iou.item():.4f}")


def main(images_dir, masks_dir, model_path):

    # Get just the first image for detailed visualization
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])[:1]

    # Initialize model and load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAEFormer(num_classes=1).to(device)
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = (
            checkpoint
            if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint
            else checkpoint.get("model_state_dict", checkpoint)
        )
        try:
            model.load_state_dict(state_dict)
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Using untrained model instead")
    else:
        print(
            f"Warning: No model weights found at {model_path}. Using untrained model."
        )

    model.eval()

    # Process single image for detailed visualization
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        print(f"\nProcessing {image_file} for detailed transformation visualization")
        image_tensor = transform(Image.open(image_path)).unsqueeze(0)
        visualize_attention_maps(model, image_tensor)
        # Also run the segmentation visualization for comparison
        mask_path = os.path.join(masks_dir, image_file)
        visualize_segmentation_with_gt(
            image_path,
            mask_path,
            model,
            save_dir=f"segmentation_results/{os.path.splitext(image_file)[0]}",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir", type=str, required=False, default="datasets/Kvasir-SEG/images"
    )
    parser.add_argument(
        "--masks_dir", type=str, required=False, default="datasets/Kvasir-SEG/masks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="model_out/kvasir_best_model.pth",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="all_visualizations/daeformer",
    )
    args = parser.parse_args()
    main(args.images_dir, args.masks_dir, args.model_path, args.save_dir)
