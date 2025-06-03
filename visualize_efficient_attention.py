import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from networks.DAEFormer import EfficientAttention
import torch.nn.functional as F


def normalize_tensor(tensor):
    """Normalize tensor to [0, 1] range for visualization"""
    tensor = tensor.detach().cpu()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    return tensor


def visualize_efficient_attention(image_tensor, save_dir="efficient_attention"):
    """Visualize the efficient attention mechanism"""
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Convert RGB to grayscale by taking mean across channels
    if image_tensor.size(1) == 3:
        image_tensor = image_tensor.mean(dim=1, keepdim=True)

    # Initialize efficient attention module with more channels for better visualization
    efficient_attn = EfficientAttention(
        in_channels=1, key_channels=8, value_channels=8, head_count=4
    )
    efficient_attn.eval()  # Set to evaluation mode

    # Process image through efficient attention
    with torch.no_grad():
        # Visualize input
        plt.figure(figsize=(8, 8))
        input_norm = normalize_tensor(image_tensor[0, 0])
        plt.imshow(input_norm, cmap="viridis")
        plt.title("Input Image (Grayscale)")
        plt.colorbar()
        plt.savefig(f"{save_dir}/input.png")
        plt.close()

        # Get attention output
        attention_output = efficient_attn(image_tensor)

        # Visualize attention output
        plt.figure(figsize=(8, 8))
        output_norm = normalize_tensor(attention_output[0, 0])
        plt.imshow(output_norm, cmap="viridis")
        plt.title("Efficient Attention Output")
        plt.colorbar()
        plt.savefig(f"{save_dir}/attention_output.png")
        plt.close()

        # Get intermediate features
        keys = efficient_attn.keys(image_tensor)  # [B, key_channels, H, W]
        queries = efficient_attn.queries(image_tensor)  # [B, key_channels, H, W]
        values = efficient_attn.values(image_tensor)  # [B, value_channels, H, W]

        # Visualize feature maps for each head
        n_heads = efficient_attn.head_count
        channels_per_head = efficient_attn.key_channels // n_heads

        # Create a grid of visualizations for each head
        for head_idx in range(n_heads):
            # Get features for this head
            start_ch = head_idx * channels_per_head
            end_ch = (head_idx + 1) * channels_per_head

            # Visualize key features for this head
            plt.figure(figsize=(15, 5))
            for i, ch in enumerate(range(start_ch, end_ch)):
                plt.subplot(1, channels_per_head, i + 1)
                key_norm = normalize_tensor(keys[0, ch])
                plt.imshow(key_norm, cmap="viridis")
                plt.title(f"Key {ch}")
                plt.axis("off")
            plt.suptitle(f"Key Features - Head {head_idx + 1}")
            plt.savefig(f"{save_dir}/keys_head_{head_idx + 1}.png")
            plt.close()

            # Visualize query features for this head
            plt.figure(figsize=(15, 5))
            for i, ch in enumerate(range(start_ch, end_ch)):
                plt.subplot(1, channels_per_head, i + 1)
                query_norm = normalize_tensor(queries[0, ch])
                plt.imshow(query_norm, cmap="viridis")
                plt.title(f"Query {ch}")
                plt.axis("off")
            plt.suptitle(f"Query Features - Head {head_idx + 1}")
            plt.savefig(f"{save_dir}/queries_head_{head_idx + 1}.png")
            plt.close()

            # Visualize value features for this head
            plt.figure(figsize=(15, 5))
            for i, ch in enumerate(range(start_ch, end_ch)):
                plt.subplot(1, channels_per_head, i + 1)
                value_norm = normalize_tensor(values[0, ch])
                plt.imshow(value_norm, cmap="viridis")
                plt.title(f"Value {ch}")
                plt.axis("off")
            plt.suptitle(f"Value Features - Head {head_idx + 1}")
            plt.savefig(f"{save_dir}/values_head_{head_idx + 1}.png")
            plt.close()

        # Compute and visualize attention weights for each head
        n, _, h, w = image_tensor.size()
        keys_reshaped = keys.reshape((n, efficient_attn.key_channels, h * w))
        queries_reshaped = queries.reshape(n, efficient_attn.key_channels, h * w)

        for head_idx in range(n_heads):
            start_ch = head_idx * channels_per_head
            end_ch = (head_idx + 1) * channels_per_head

            # Get key and query features for this head
            head_keys = keys_reshaped[
                :, start_ch:end_ch, :
            ]  # [B, channels_per_head, H*W]
            head_queries = queries_reshaped[
                :, start_ch:end_ch, :
            ]  # [B, channels_per_head, H*W]

            # Compute attention weights for this head
            attention_weights = torch.bmm(
                head_queries.transpose(1, 2), head_keys
            )  # [B, H*W, H*W]
            attention_weights = F.softmax(attention_weights, dim=-1)

            # Visualize attention weights for a few selected positions
            n_positions = 4
            positions = [
                (h // 4, w // 4),
                (h // 4, 3 * w // 4),
                (3 * h // 4, w // 4),
                (3 * h // 4, 3 * w // 4),
            ]

            plt.figure(figsize=(15, 15))
            for i, (y, x) in enumerate(positions):
                pos_idx = y * w + x
                plt.subplot(2, 2, i + 1)
                weights = attention_weights[0, pos_idx].reshape(h, w)
                weights_norm = normalize_tensor(weights)
                plt.imshow(weights_norm, cmap="viridis")
                plt.title(f"Attention at ({y}, {x})")
                plt.colorbar()
            plt.suptitle(f"Attention Weights - Head {head_idx + 1}")
            plt.savefig(f"{save_dir}/attention_weights_head_{head_idx + 1}.png")
            plt.close()

        # Create a summary visualization
        plt.figure(figsize=(20, 10))

        # Input
        plt.subplot(231)
        plt.imshow(input_norm, cmap="viridis")
        plt.title("Input Image")
        plt.axis("off")

        # Output
        plt.subplot(232)
        plt.imshow(output_norm, cmap="viridis")
        plt.title("Attention Output")
        plt.axis("off")

        # Average attention weights across heads
        avg_attention = torch.zeros(h, w, device=image_tensor.device)
        for head_idx in range(n_heads):
            start_ch = head_idx * channels_per_head
            end_ch = (head_idx + 1) * channels_per_head
            head_keys = keys_reshaped[:, start_ch:end_ch, :]
            head_queries = queries_reshaped[:, start_ch:end_ch, :]
            attention_weights = torch.bmm(head_queries.transpose(1, 2), head_keys)
            attention_weights = F.softmax(attention_weights, dim=-1)
            # Take attention weights for center position
            center_pos = (h * w) // 2
            avg_attention += attention_weights[0, center_pos].reshape(h, w)
        avg_attention /= n_heads
        avg_attention_norm = normalize_tensor(avg_attention)

        plt.subplot(233)
        plt.imshow(avg_attention_norm, cmap="viridis")
        plt.title("Average Attention (Center)")
        plt.axis("off")

        # Feature maps for first head
        plt.subplot(234)
        key_norm = normalize_tensor(keys[0, 0])
        plt.imshow(key_norm, cmap="viridis")
        plt.title("Key Features (First Head)")
        plt.axis("off")

        plt.subplot(235)
        query_norm = normalize_tensor(queries[0, 0])
        plt.imshow(query_norm, cmap="viridis")
        plt.title("Query Features (First Head)")
        plt.axis("off")

        plt.subplot(236)
        value_norm = normalize_tensor(values[0, 0])
        plt.imshow(value_norm, cmap="viridis")
        plt.title("Value Features (First Head)")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/attention_summary.png")
        plt.close()


def main():
    # Load and preprocess image
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Load sample image (replace with your image path)
    image = Image.open("sample.jpg")
    image_tensor = transform(image).unsqueeze(0)

    # Visualize efficient attention
    visualize_efficient_attention(image_tensor)


if __name__ == "__main__":
    main()
