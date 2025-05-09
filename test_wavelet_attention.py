import torch
from networks.DAEFormer import WaveletAttention


def test_wavelet_attention():
    # Test parameters
    batch_size = 2
    channels = 3
    height = 32
    width = 32

    # Create input tensor
    x = torch.randn(batch_size, channels, height, width)

    # Initialize WaveletAttention
    wavelet_attn = WaveletAttention(in_channels=channels, wavelet="db1", level=1)

    try:
        # Forward pass
        output = wavelet_attn(x)

        # Check output shape
        assert (
            output.shape == x.shape
        ), f"Output shape {output.shape} doesn't match input shape {x.shape}"

        # Check if output contains NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        # Check if output is different from input (attention is working)
        assert not torch.allclose(
            output, x
        ), "Output is identical to input, attention might not be working"

        print("All tests passed!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise e


if __name__ == "__main__":
    test_wavelet_attention()
