import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from visualize_daeformer import visualize_attention_maps
from visualize_wavelet_attention import visualize_wavelet_attention
from visualize_efficient_attention import visualize_efficient_attention
from networks.DAEFormer import DAEFormer as DAEFormerOld
from networks.DAEFormer_new import DAEFormer as DAEFormerNew
import torch


def create_comparison_grid(
    image_path, save_dir="all_visualizations", model_path="model_out/best_model.pth"
):
    """Create a comprehensive comparison of all visualizations"""
    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess image
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_old = DAEFormerOld(num_classes=9).to(device)
    model_new = DAEFormerNew(num_classes=9).to(device)

    # Load trained weights
    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Handle both direct state dict and dictionary-wrapped state dict
        state_dict = (
            checkpoint
            if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint
            else checkpoint.get("model_state_dict", checkpoint)
        )

        # Try to load weights for both models
        try:
            model_old.load_state_dict(state_dict)
            print("Successfully loaded weights for old model")
        except Exception as e:
            print(f"Could not load weights for old model: {e}")

        try:
            model_new.load_state_dict(state_dict)
            print("Successfully loaded weights for new model")
        except Exception as e:
            print(f"Could not load weights for new model: {e}")
    else:
        print(
            f"Warning: No model weights found at {model_path}. Using randomly initialized models."
        )

    model_old.eval()
    model_new.eval()

    # Run all visualizations
    print("Running DAEFormer visualizations...")
    visualize_attention_maps(model_old, image_tensor, save_dir=f"{save_dir}/daeformer")

    print("Running Wavelet Attention visualizations...")
    visualize_wavelet_attention(image_tensor, save_dir=f"{save_dir}/wavelet")

    print("Running Efficient Attention visualizations...")
    visualize_efficient_attention(image_tensor, save_dir=f"{save_dir}/efficient")

    # Create comparison grid
    plt.figure(figsize=(20, 15))

    # Original image
    plt.subplot(331)
    plt.imshow(image_tensor[0].permute(1, 2, 0).cpu().numpy())
    plt.title("Input Image")
    plt.axis("off")

    # Load and display visualizations
    visualization_paths = {
        "Efficient Attention": f"{save_dir}/efficient/attention_output.png",
        "Wavelet Attention": f"{save_dir}/wavelet/attention_output.png",
        "Encoder Stage 1": f"{save_dir}/daeformer/2_encoder_stage_1.png",
        "Encoder Stage 2": f"{save_dir}/daeformer/2_encoder_stage_2.png",
        "Encoder Stage 3": f"{save_dir}/daeformer/2_encoder_stage_3.png",
        "Decoder Stage 1": f"{save_dir}/daeformer/3_decoder_stage_1.png",
        "Decoder Stage 2": f"{save_dir}/daeformer/3_decoder_stage_2.png",
        "Final Output": f"{save_dir}/daeformer/4_final_output.png",
    }

    for i, (title, path) in enumerate(visualization_paths.items(), 2):
        if os.path.exists(path):
            img = plt.imread(path)
            plt.subplot(3, 3, i)
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_grid.png")
    plt.close()

    print(f"All visualizations saved to {save_dir}")
    print("Comparison grid saved as comparison_grid.png")


def main():
    # Replace with your image path
    import sys

    image_path = sys.argv[1]
    create_comparison_grid(image_path)


if __name__ == "__main__":
    main()
