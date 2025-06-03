import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from networks.DAEFormer import DAEFormer as DAEFormerOld
from networks.DAEFormer_new import DAEFormer as DAEFormerNew

def compare_architectures(image_path, save_dir='comparison_results'):
    """Compare old and new DAEFormer architectures"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    
    # Initialize both models
    model_old = DAEFormerOld(num_classes=9)
    model_new = DAEFormerNew(num_classes=9)
    model_old.eval()
    model_new.eval()
    
    # Process through both models
    with torch.no_grad():
        # Old model
        output_old = model_old(image_tensor)
        
        # New model
        output_new = model_new(image_tensor)
        
        # Visualize results
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(image_tensor[0].permute(1, 2, 0).cpu().numpy())
        plt.title('Input Image')
        plt.axis('off')
        
        # Old model output
        plt.subplot(132)
        plt.imshow(output_old[0, 0].cpu().numpy(), cmap='viridis')
        plt.title('Old DAEFormer Output')
        plt.colorbar()
        plt.axis('off')
        
        # New model output
        plt.subplot(133)
        plt.imshow(output_new[0, 0].cpu().numpy(), cmap='viridis')
        plt.title('New DAEFormer Output')
        plt.colorbar()
        plt.axis('off')
        
        plt.savefig(f'{save_dir}/comparison.png')
        plt.close()
        
        # Save individual outputs
        plt.figure(figsize=(8, 8))
        plt.imshow(output_old[0, 0].cpu().numpy(), cmap='viridis')
        plt.title('Old DAEFormer Output')
        plt.colorbar()
        plt.savefig(f'{save_dir}/old_output.png')
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(output_new[0, 0].cpu().numpy(), cmap='viridis')
        plt.title('New DAEFormer Output')
        plt.colorbar()
        plt.savefig(f'{save_dir}/new_output.png')
        plt.close()

def main():
    # Replace with your image path
    import sys
    image_path = sys.argv[1]
    compare_architectures(image_path)

if __name__ == '__main__':
    main() 