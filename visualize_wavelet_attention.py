import torch
import numpy as np
import matplotlib.pyplot as plt
import pywt
from networks.DAEFormer import WaveletAttention


def plot_wavelet_coeffs(coeffs, title):
    """Plot wavelet coefficients."""
    # Get the number of levels
    n_levels = len(coeffs) - 1

    # Create a figure
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle(title, fontsize=14)

    # Plot approximation coefficients
    ax = fig.add_subplot(n_levels + 1, 3, 1)
    ax.imshow(coeffs[0], cmap="viridis")
    ax.set_title("Approximation")
    ax.axis("off")

    # Plot detail coefficients
    for level in range(n_levels):
        for i, direction in enumerate(["Horizontal", "Vertical", "Diagonal"]):
            ax = fig.add_subplot(n_levels + 1, 3, (level + 1) * 3 + i + 1)
            ax.imshow(coeffs[level + 1][i], cmap="viridis")
            ax.set_title(f"Level {level + 1} {direction}")
            ax.axis("off")

    plt.tight_layout()
    return fig


def visualize_attention_effects(input_tensor, wavelet_attn):
    """Visualize the effects of wavelet attention on a single channel."""
    # Get a single image and channel for visualization
    image = input_tensor[0, 0].detach().cpu().numpy()

    # Get wavelet coefficients of input
    input_coeffs = pywt.wavedec2(image, wavelet_attn.wavelet, level=wavelet_attn.level)

    # Apply attention
    with torch.no_grad():
        output = wavelet_attn(input_tensor)
    output_image = output[0, 0].cpu().numpy()

    # Get wavelet coefficients of output
    output_coeffs = pywt.wavedec2(
        output_image, wavelet_attn.wavelet, level=wavelet_attn.level
    )

    # Plot original and processed images
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(image, cmap="viridis")
    plt.title("Original Image")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(output_image, cmap="viridis")
    plt.title("After Wavelet Attention")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Plot wavelet coefficients
    plot_wavelet_coeffs(input_coeffs, "Input Wavelet Coefficients")
    plt.show()

    plot_wavelet_coeffs(output_coeffs, "Output Wavelet Coefficients")
    plt.show()

    # Plot attention weights
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.bar(["Alpha", "Beta"], [wavelet_attn.alpha.item(), wavelet_attn.beta.item()])
    plt.title("Learned Attention Weights")

    # Plot difference map
    plt.subplot(122)
    diff = output_image - image
    plt.imshow(diff, cmap="RdBu")
    plt.title("Difference Map")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create sample input
    batch_size = 1
    channels = 1
    height = 32
    width = 32

    # Create a test pattern
    x = torch.zeros(batch_size, channels, height, width)
    # Add some patterns for better visualization
    for i in range(height):
        for j in range(width):
            x[0, 0, i, j] = np.sin(i / 4) + np.cos(j / 4)

    # Initialize WaveletAttention
    wavelet_attn = WaveletAttention(in_channels=channels, wavelet="db1", level=2)

    # Visualize
    visualize_attention_effects(x, wavelet_attn)
