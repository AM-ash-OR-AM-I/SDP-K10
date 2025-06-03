import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from networks.DAEFormer import DAEFormer

def visualize_tensor(tensor, title, save_path=None):
    """Visualize a tensor as an image"""
    # Convert tensor to numpy array
    if tensor.dim() == 4:  # [B, C, H, W]
        tensor = tensor[0]  # Take first image in batch
    
    # If tensor has multiple channels, take mean across channels
    if tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0)
    
    # Normalize to [0, 1]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    plt.figure(figsize=(8, 8))
    plt.imshow(tensor.detach().cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_attention_maps(model, image_tensor, save_dir='attention_maps'):
    """Visualize attention maps at different stages"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Initial Image
    visualize_tensor(image_tensor, 'Input Image', f'{save_dir}/1_input.png')
    
    # 2. Encoder Stages
    with torch.no_grad():
        # Get encoder outputs
        if image_tensor.size(1) == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)
        
        # Get encoder outputs
        encoder_outputs = model.backbone(image_tensor)
        
        # Visualize each encoder stage
        for i, output in enumerate(encoder_outputs):
            visualize_tensor(output, f'Encoder Stage {i+1}', f'{save_dir}/2_encoder_stage_{i+1}.png')
        
        # 3. Decoder Stages
        b, c, _, _ = encoder_outputs[2].shape
        
        # Decoder Stage 1
        tmp_2 = model.decoder_2(encoder_outputs[2].permute(0, 2, 3, 1).view(b, -1, c))
        visualize_tensor(tmp_2.view(b, -1, 28, 28).permute(0, 3, 1, 2), 
                        'Decoder Stage 1', f'{save_dir}/3_decoder_stage_1.png')
        
        # Decoder Stage 2
        tmp_1 = model.decoder_1(tmp_2, encoder_outputs[1].permute(0, 2, 3, 1))
        visualize_tensor(tmp_1.view(b, -1, 56, 56).permute(0, 3, 1, 2), 
                        'Decoder Stage 2', f'{save_dir}/3_decoder_stage_2.png')
        
        # Decoder Stage 3 (Final Output)
        tmp_0 = model.decoder_0(tmp_1, encoder_outputs[0].permute(0, 2, 3, 1))
        visualize_tensor(tmp_0, 'Final Segmentation Output', f'{save_dir}/4_final_output.png')

def main():
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load sample image (replace with your image path)
    image = Image.open('datasets/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg')
    image_tensor = transform(image).unsqueeze(0)
    
    # Initialize model
    model = DAEFormer(num_classes=9)
    model.eval()
    
    # Visualize attention maps
    visualize_attention_maps(model, image_tensor)

if __name__ == '__main__':
    main() 