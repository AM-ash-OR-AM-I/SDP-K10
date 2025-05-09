import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import pywt

from networks.segformer import *


class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width

        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(
                keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2
            )
            query = F.softmax(
                queries[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )
            value = values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(
            B, D, self.height, self.width
        )
        reprojected_value = (
            self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        )
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value


class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(
        self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp="mix"
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.attn = Cross_Attention(
            key_dim, value_dim, height, width, head_count=head_count
        )
        self.norm2 = nn.LayerNorm((in_dim * 2))
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)

        attn = self.attn(norm_1, norm_2)
        # attn = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(attn)

        # residual1 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x1)
        # residual2 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x2)
        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(
                keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2
            )

            query = F.softmax(
                queries[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )

            value = values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(
                n, head_value_channels, h, w
            )  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class WaveletAttention(nn.Module):
    """
    Wavelet-based attention module that uses wavelet transform for spatial attention
    Input -> x:[B, D, H, W]
    Output -> [B, D, H, W]
    """

    def __init__(self, in_channels, wavelet="db1", level=1):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet = wavelet
        self.level = level

        # Learnable parameters for wavelet coefficients
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        # Convolution layers for processing wavelet coefficients
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Process each channel separately
        output = []
        for b in range(B):
            channel_outputs = []
            for c in range(C):
                # Convert to numpy for wavelet transform
                signal = x[b, c].detach().cpu().numpy()

                # Perform wavelet transform
                coeffs = pywt.wavedec2(signal, self.wavelet, level=self.level)

                # Process approximation coefficients
                cA = torch.from_numpy(coeffs[0]).to(x.device)
                cA = cA.unsqueeze(0).unsqueeze(0)  # [1, 1, H', W']
                cA = self.conv1(cA)

                # Process detail coefficients
                detail_coeffs = []
                for i in range(1, len(coeffs)):
                    level_details = []
                    for j in range(3):  # Horizontal, vertical, and diagonal details
                        detail = torch.from_numpy(coeffs[i][j]).to(x.device)
                        detail = detail.unsqueeze(0).unsqueeze(0)  # [1, 1, H', W']
                        detail = self.conv2(detail)
                        # Resize detail to match approximation size
                        if detail.shape != cA.shape:
                            detail = F.interpolate(
                                detail,
                                size=cA.shape[2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        level_details.append(detail)
                    detail_coeffs.extend(level_details)

                # Sum all detail coefficients (now they have the same size)
                detail_sum = sum(detail_coeffs)

                # Combine coefficients with learnable weights
                combined = self.alpha * cA + self.beta * detail_sum

                # Prepare coefficients for inverse transform
                processed_coeffs = [combined.detach().squeeze().cpu().numpy()]

                # Reconstruct detail coefficients for each level
                detail_idx = 0
                for i in range(1, len(coeffs)):
                    level_coeffs = []
                    for j in range(3):
                        if detail_idx < len(detail_coeffs):
                            # Resize back to original detail coefficient size
                            detail = F.interpolate(
                                detail_coeffs[detail_idx],
                                size=coeffs[i][j].shape,
                                mode="bilinear",
                                align_corners=False,
                            )
                            detail = detail.detach().squeeze().cpu().numpy()
                            level_coeffs.append(detail)
                            detail_idx += 1
                        else:
                            level_coeffs.append(coeffs[i][j])
                    processed_coeffs.append(tuple(level_coeffs))

                # Perform inverse wavelet transform
                reconstructed = pywt.waverec2(processed_coeffs, self.wavelet)

                # Ensure the output has the same size as input
                if reconstructed.shape != (H, W):
                    reconstructed = reconstructed[:H, :W]

                # Convert back to tensor
                reconstructed = torch.from_numpy(reconstructed).to(x.device)
                reconstructed = reconstructed.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                channel_outputs.append(reconstructed)

            # Combine channel outputs
            channel_output = torch.cat(channel_outputs, dim=1)  # [1, C, H, W]
            output.append(channel_output)

        # Combine batch outputs
        output = torch.cat(output, dim=0)  # [B, C, H, W]

        return output


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = WaveletAttention(in_channels=in_dim, wavelet="db1", level=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # dual attention structure, wavelet attention first then channel attention
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)

        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)

        mx = add3 + mlp2
        return mx


# Encoder
class MiT(nn.Module):
    def __init__(
        self,
        image_size,
        in_dim,
        key_dim,
        value_dim,
        layers,
        head_count=1,
        token_mlp="mix_skip",
    ):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4,
            patch_sizes[1],
            strides[1],
            padding_sizes[1],
            in_dim[0],
            in_dim[1],
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8,
            patch_sizes[2],
            strides[2],
            padding_sizes[2],
            in_dim[1],
            in_dim[2],
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [
                DualTransformerBlock(
                    in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp
                )
                for _ in range(layers[0])
            ]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [
                DualTransformerBlock(
                    in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp
                )
                for _ in range(layers[1])
            ]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [
                DualTransformerBlock(
                    in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp
                )
                for _ in range(layers[2])
            ]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


# Decoder
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = (
            nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=C // (self.dim_scale**2),
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
        self,
        input_size,
        in_out_chan,
        head_count,
        token_mlp_mode,
        n_class=9,
        norm_layer=nn.LayerNorm,
        is_last=False,
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims,
                key_dim,
                value_dim,
                input_size[0],
                input_size[1],
                head_count,
                token_mlp_mode,
            )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(
                input_resolution=input_size,
                dim=out_dim,
                dim_scale=2,
                norm_layer=norm_layer,
            )
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims * 2,
                key_dim,
                value_dim,
                input_size[0],
                input_size[1],
                head_count,
                token_mlp_mode,
            )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size,
                dim=out_dim,
                dim_scale=4,
                norm_layer=norm_layer,
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = DualTransformerBlock(
            out_dim, key_dim, value_dim, head_count, token_mlp_mode
        )
        self.layer_former_2 = DualTransformerBlock(
            out_dim, key_dim, value_dim, head_count, token_mlp_mode
        )

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(
                    self.layer_up(tran_layer_2)
                    .view(b, 4 * h, 4 * w, -1)
                    .permute(0, 3, 1, 2)
                )
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


class DAEFormer(nn.Module):
    def __init__(self, num_classes=9, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        dims, key_dim, value_dim, layers = [
            [128, 320, 512],
            [128, 320, 512],
            [128, 320, 512],
            [2, 2, 2],
        ]
        self.backbone = MiT(
            image_size=224,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
        )

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [64, 128, 128, 128, 160],
            [320, 320, 320, 320, 256],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc = self.backbone(x)

        b, c, _, _ = output_enc[2].shape

        # ---------------Decoder-------------------------
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))

        return tmp_0
