import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import pywt
from networks.DAEFormer_new import WaveletAttention


def visualize_wavelet_decomposition(
    image_tensor, wavelet="db1", level=1, save_dir="wavelet_decomposition"
):
    """Visualize wavelet decomposition of an image"""
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Convert tensor to numpy
    image = image_tensor[0, 0].detach().cpu().numpy()

    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # Visualize approximation coefficients
    plt.figure(figsize=(8, 8))
    plt.imshow(coeffs[0], cmap="viridis")
    plt.title("Approximation Coefficients")
    plt.colorbar()
    plt.savefig(f"{save_dir}/approximation.png")
    plt.close()

    # Visualize detail coefficients for each level
    for i in range(1, len(coeffs)):
        plt.figure(figsize=(15, 5))
        for j, (title, detail) in enumerate(
            zip(["Horizontal", "Vertical", "Diagonal"], coeffs[i])
        ):
            plt.subplot(1, 3, j + 1)
            plt.imshow(detail, cmap="viridis")
            plt.title(f"{title} Details - Level {i}")
            plt.colorbar()
        plt.savefig(f"{save_dir}/details_level_{i}.png")
        plt.close()


def visualize_wavelet_attention(image_tensor, save_dir="wavelet_attention"):
    """Visualize the wavelet attention mechanism"""
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Initialize wavelet attention module
    wavelet_attn = WaveletAttention(in_channels=1, wavelet="db1", level=1)

    # Process image through wavelet attention
    with torch.no_grad():
        # Visualize input
        plt.figure(figsize=(8, 8))
        plt.imshow(image_tensor[0, 0].detach().cpu().numpy(), cmap="viridis")
        plt.title("Input Image")
        plt.colorbar()
        plt.savefig(f"{save_dir}/input.png")
        plt.close()

        # Get attention output
        attention_output = wavelet_attn(image_tensor)

        # Visualize attention output
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_output[0, 0].detach().cpu().numpy(), cmap="viridis")
        plt.title("Wavelet Attention Output")
        plt.colorbar()
        plt.savefig(f"{save_dir}/attention_output.png")
        plt.close()

        # Visualize wavelet decomposition
        visualize_wavelet_decomposition(
            image_tensor, save_dir=f"{save_dir}/decomposition"
        )


def main():
    # Load and preprocess image
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Load sample image (replace with your image path)
    image = Image.open("sample.jpg")
    image_tensor = transform(image).unsqueeze(0)

    # Visualize wavelet attention
    visualize_wavelet_attention(image_tensor)


if __name__ == "__main__":
    main()
