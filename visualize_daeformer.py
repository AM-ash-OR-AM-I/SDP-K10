import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from networks.DAEFormer import DAEFormer
from networks.DAEFormer_new import DAEFormer as DAEFormer_new
import os


def visualize_tensor(
    tensor, title, save_path=None, cmap="viridis", colorbar_label=None
):
    """Visualize a tensor as an image with enhanced visualization options"""
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
    im = plt.imshow(tensor.detach().cpu().numpy(), cmap=cmap)
    plt.title(title)
    if colorbar_label:
        plt.colorbar(im, label=colorbar_label)
    else:
        plt.colorbar(im)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_attention_maps(
    model, image_tensor, save_dir="all_visualizations/daeformer"
):
    """Visualize the complete transformation process of DAEFormer"""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Initial Image
    visualize_tensor(
        image_tensor,
        "1. Input Image",
        f"{save_dir}/1_input.png",
        cmap="gray" if image_tensor.size(1) == 1 else None,
    )

    with torch.no_grad():
        # Get encoder outputs
        if image_tensor.size(1) == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        # Get encoder outputs
        encoder_outputs = model.backbone(image_tensor)

        # Visualize each encoder stage with more details
        for i, output in enumerate(encoder_outputs):
            b, c, h, w = output.shape
            visualize_tensor(
                output,
                f"2. Encoder Stage {i+1}\nShape: {c} channels, {h}x{w} spatial",
                f"{save_dir}/2_encoder_stage_{i+1}.png",
                cmap="viridis",
                colorbar_label="Activation Strength",
            )

        # 3. Decoder Stages
        b, c, _, _ = encoder_outputs[2].shape

        # Decoder Stage 1
        tmp_2 = model.decoder_2(encoder_outputs[2].permute(0, 2, 3, 1).view(b, -1, c))
        decoder_1_output = tmp_2.view(b, 28, 28, -1).permute(0, 3, 1, 2)
        visualize_tensor(
            decoder_1_output,
            "3. Decoder Stage 1\nUpsampling from 14x14 to 28x28",
            f"{save_dir}/3_decoder_stage_1.png",
            cmap="viridis",
            colorbar_label="Activation Strength",
        )

        # Decoder Stage 2
        tmp_1 = model.decoder_1(tmp_2, encoder_outputs[1].permute(0, 2, 3, 1))
        decoder_2_output = tmp_1.view(b, 56, 56, -1).permute(0, 3, 1, 2)
        visualize_tensor(
            decoder_2_output,
            "3. Decoder Stage 2\nUpsampling from 28x28 to 56x56",
            f"{save_dir}/3_decoder_stage_2.png",
            cmap="viridis",
            colorbar_label="Activation Strength",
        )

        # Decoder Stage 3 (Final Output)
        tmp_0 = model.decoder_0(tmp_1, encoder_outputs[0].permute(0, 2, 3, 1))

        # Show raw decoder output (before sigmoid)
        visualize_tensor(
            tmp_0[0:1, 0:1],  # Take first channel
            "4. Decoder Stage 3 (Raw Output)\nBefore Sigmoid Activation",
            f"{save_dir}/4_decoder_stage_3_raw.png",
            cmap="viridis",
            colorbar_label="Raw Activation",
        )

        # Show final segmentation mask (after sigmoid)
        final_output = torch.sigmoid(tmp_0[0:1, 0:1])
        visualize_tensor(
            final_output,
            "4. Final Output\nSegmentation Mask (After Sigmoid)",
            f"{save_dir}/4_final_output.png",
            cmap="gray",
            colorbar_label="Probability",
        )

        # Show final binary mask (thresholded at 0.5)
        binary_mask = (final_output > 0.5).float()
        visualize_tensor(
            binary_mask,
            "5. Final Binary Mask\nThresholded at 0.5",
            f"{save_dir}/5_final_binary_mask.png",
            cmap="gray",
            colorbar_label="Mask (0=bg, 1=fg)",
        )

        print(f"Transformation visualizations saved to {save_dir}")
        print("Visualization sequence:")
        print("1. Input Image")
        print("2. Encoder Stages (3 stages, progressively smaller spatial dimensions)")
        print("3. Decoder Stages (2 stages, progressively larger spatial dimensions)")
        print("4. Decoder Stage 3 (Raw output before sigmoid)")
        print("4. Final Output (Segmentation mask after sigmoid)")
        print("5. Final Binary Mask (Thresholded at 0.5)")


def multiclass_dice(pred, target, num_classes):
    dice = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice.append((2.0 * intersection) / (union + 1e-8))
    return float(torch.mean(torch.tensor(dice)))


def multiclass_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        ious.append(intersection / (union + 1e-8))
    return float(torch.mean(torch.tensor(ious)))


def visualize_segmentation_with_gt(
    image_path, mask_path, model, save_dir="segmentation_results", num_classes=1
):
    """Visualize segmentation results with ground truth comparison for a single image
    Returns: dice, iou
    """
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

    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        if num_classes > 1:
            # Multi-class: Synapse
            output = torch.softmax(output, dim=1)
            pred_bin = output.argmax(dim=1, keepdim=True)
            mask_bin = mask_tensor.long()  # Use integer labels as is
        else:
            # Binary: Kvasir
            output = torch.sigmoid(output)
            pred_bin = (output > 0.5).float()
            mask_bin = (mask_tensor > 0.5).float()

    # Only consider the foreground (mask == 1 or pred == 1) for binary
    if num_classes > 1:
        mask_fg = mask_bin[0, 0]
        pred_fg = pred_bin[0, 0]
        dice = multiclass_dice(pred_fg, mask_fg, num_classes)
        iou = multiclass_iou(pred_fg, mask_fg, num_classes)
    else:
        mask_fg = mask_bin[0, 0]
        pred_fg = pred_bin[0, 0]
        intersection = (pred_fg * mask_fg).sum()
        union = pred_fg.sum() + mask_fg.sum() - intersection
        dice = (2.0 * intersection) / (pred_fg.sum() + mask_fg.sum() + 1e-8)
        iou = intersection / (union + 1e-8)

    # Debug prints
    print("Mask sum (foreground pixels):", mask_fg.sum().item())
    print("Prediction sum (foreground pixels):", pred_fg.sum().item())
    print("Dice Score:", dice)
    print("IoU Score:", iou)

    # Create visualization
    plt.figure(figsize=(15, 6))

    # Input image
    plt.subplot(131)
    plt.imshow(
        image_tensor[0].permute(1, 2, 0).cpu().numpy(),
        cmap=None if num_classes == 1 else "gray",
    )
    plt.title("Input Image")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(132)
    if num_classes > 1:
        plt.imshow(mask_fg.cpu().numpy(), cmap="tab10", vmin=0, vmax=num_classes - 1)
    else:
        plt.imshow(mask_fg.cpu().numpy(), cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Model prediction
    plt.subplot(133)
    if num_classes > 1:
        plt.imshow(pred_fg.cpu().numpy(), cmap="tab10", vmin=0, vmax=num_classes - 1)
        plt.title(f"Model Prediction\nMean Dice: {dice:.3f}, Mean IoU: {iou:.3f}")
    else:
        plt.imshow(pred_fg.cpu().numpy(), cmap="gray")
        plt.title(f"Model Prediction\nDice: {dice.item():.3f}, IoU: {iou.item():.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/segmentation_comparison.png")
    plt.close()

    # Create overlay visualization
    plt.figure(figsize=(10, 10))
    if num_classes > 1:
        # For multi-class, overlay not as meaningful, but show prediction vs. GT
        plt.subplot(121)
        plt.imshow(mask_fg.cpu().numpy(), cmap="tab10", vmin=0, vmax=num_classes - 1)
        plt.title("Ground Truth Mask")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(pred_fg.cpu().numpy(), cmap="tab10", vmin=0, vmax=num_classes - 1)
        plt.title("Model Prediction")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/segmentation_overlay.png")
    else:
        # Convert tensors to numpy arrays
        pred_np = pred_fg.cpu().numpy()
        gt_np = mask_fg.cpu().numpy()
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
    print(f"Dice Score: {dice:.4f}")
    print(f"IoU Score: {iou:.4f}")

    return dice, iou


def main(images_dir, masks_dir, model_path, save_dir, model_type="original"):
    # Define the transform
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Get first 30 images
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]
    )[:30]

    # Initialize model and load trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "synapse" in model_path.lower():
        num_classes = 9
    else:
        num_classes = 1

    # Initialize the appropriate model based on model_type
    if model_type == "new":
        model = DAEFormer_new(num_classes=num_classes).to(device)
    else:
        model = DAEFormer(num_classes=num_classes).to(device)

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

    # For summary
    dice_scores = []
    iou_scores = []
    image_names = []

    # Create model-specific subdirectory
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        print(f"\nProcessing {image_file} for detailed transformation visualization")

        # Load and transform image
        try:
            image = Image.open(image_path).convert("RGB")  # Ensure RGB format
            image_tensor = transform(image).unsqueeze(0)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue

        # Use folder structure: <save_dir>/<model_type>/<image_basename>/attention_maps/ and .../segmentation_results/
        image_base = os.path.splitext(image_file)[0]
        parent_dir = os.path.join(model_save_dir, image_base)
        attention_save_dir = os.path.join(parent_dir, "attention_maps")
        segmentation_save_dir = os.path.join(parent_dir, "segmentation_results")

        visualize_attention_maps(
            model,
            image_tensor,
            save_dir=attention_save_dir,
        )
        # Also run the segmentation visualization for comparison
        mask_path = os.path.join(masks_dir, image_file)
        dice, iou = visualize_segmentation_with_gt(
            image_path,
            mask_path,
            model,
            save_dir=segmentation_save_dir,
            num_classes=num_classes,
        )
        dice_scores.append(dice)
        iou_scores.append(iou)
        image_names.append(image_base)

    # Plot summary bar graph
    if len(image_names) > 0:
        x = np.arange(len(image_names))
        width = 0.35
        plt.figure(figsize=(max(10, len(image_names) * 0.4), 6))
        plt.bar(x - width / 2, dice_scores, width, label="Dice")
        plt.bar(x + width / 2, iou_scores, width, label="IoU")
        plt.xlabel("Image")
        plt.ylabel("Score")
        plt.title(f"Dice and IoU Scores for Each Image ({model_type} model)")
        plt.xticks(x, image_names, rotation=90, fontsize=8)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_save_dir, "summary_metrics.png"))
        plt.close()
        print(
            f"Summary metrics plot saved to {os.path.join(model_save_dir, 'summary_metrics.png')}"
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
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="original",
        choices=["original", "new"],
        help="Type of model being used (original or new)",
    )
    args = parser.parse_args()
    main(
        args.images_dir, args.masks_dir, args.model_path, args.save_dir, args.model_type
    )
