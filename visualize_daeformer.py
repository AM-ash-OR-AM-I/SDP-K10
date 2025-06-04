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


def visualize_attention_maps(model, image_tensor, save_dir="attention_maps"):
    """Visualize attention maps at different stages"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Initial Image
    visualize_tensor(image_tensor, "Input Image", f"{save_dir}/1_input.png")

    # 2. Encoder Stages
    with torch.no_grad():
        # Get encoder outputs
        if image_tensor.size(1) == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        # Get encoder outputs
        encoder_outputs = model.backbone(image_tensor)

        # Visualize each encoder stage
        for i, output in enumerate(encoder_outputs):
            visualize_tensor(
                output, f"Encoder Stage {i+1}", f"{save_dir}/2_encoder_stage_{i+1}.png"
            )

        # 3. Decoder Stages
        b, c, _, _ = encoder_outputs[2].shape

        # Decoder Stage 1
        tmp_2 = model.decoder_2(encoder_outputs[2].permute(0, 2, 3, 1).view(b, -1, c))
        visualize_tensor(
            tmp_2.view(b, -1, 28, 28).permute(0, 3, 1, 2),
            "Decoder Stage 1",
            f"{save_dir}/3_decoder_stage_1.png",
        )

        # Decoder Stage 2
        tmp_1 = model.decoder_1(tmp_2, encoder_outputs[1].permute(0, 2, 3, 1))
        visualize_tensor(
            tmp_1.view(b, -1, 56, 56).permute(0, 3, 1, 2),
            "Decoder Stage 2",
            f"{save_dir}/3_decoder_stage_2.png",
        )

        # Decoder Stage 3 (Final Output)
        tmp_0 = model.decoder_0(tmp_1, encoder_outputs[0].permute(0, 2, 3, 1))
        visualize_tensor(
            tmp_0, "Final Segmentation Output", f"{save_dir}/4_final_output.png"
        )


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


def main():
    # Directory paths
    images_dir = "datasets/Kvasir-SEG/images"
    masks_dir = "datasets/Kvasir-SEG/masks"

    # Get the first 10 image filenames (assuming .jpg extension)
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])[:10]

    # Initialize model and load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DAEFormer(num_classes=1).to(device)

    model_path = "model_out/kvasir_best_model.pth"
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

    # Visualize segmentation with ground truth for 10 images
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file)
        print(f"\nProcessing {image_file} ({idx+1}/10)")
        visualize_segmentation_with_gt(
            image_path,
            mask_path,
            model,
            save_dir=f"segmentation_results/{os.path.splitext(image_file)[0]}",
        )


if __name__ == "__main__":
    main()
