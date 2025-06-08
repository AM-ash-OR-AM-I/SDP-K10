import os
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paths
BASE_DIR = "all_visualizations/daeformer"
MODEL_TYPES = ["original", "new"]
RESULT_SUBDIR = "segmentation_results"
OVERLAY_FILENAME = "segmentation_overlay.png"
OUTPUT_GIF = "segmentation_comparison.gif"

# Find all image names present in both model folders
original_dir = os.path.join(BASE_DIR, MODEL_TYPES[0])
new_dir = os.path.join(BASE_DIR, MODEL_TYPES[1])

original_images = set(os.listdir(original_dir))
new_images = set(os.listdir(new_dir))
common_images = sorted(list(original_images & new_images))

frames = []

for image_name in common_images:
    orig_overlay_path = os.path.join(
        original_dir, image_name, RESULT_SUBDIR, OVERLAY_FILENAME
    )
    new_overlay_path = os.path.join(
        new_dir, image_name, RESULT_SUBDIR, OVERLAY_FILENAME
    )
    if not (os.path.exists(orig_overlay_path) and os.path.exists(new_overlay_path)):
        continue
    # Load images
    orig_img = Image.open(orig_overlay_path).convert("RGB")
    new_img = Image.open(new_overlay_path).convert("RGB")
    # Resize to same size if needed
    if orig_img.size != new_img.size:
        new_img = new_img.resize(orig_img.size)

    # Add labels to each
    def add_label(img, label):
        img = img.copy()
        draw = ImageDraw.Draw(img)
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default(size=24)
        bar_height = 40
        # Draw white rectangle at the bottom
        draw.rectangle(
            [0, img.height - bar_height, img.width, img.height],
            fill=(255, 255, 255, 230),
        )
        # Center the label
        try:
            # Pillow >= 8.0.0
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # Older Pillow
            text_w, text_h = font.getsize(label)
        text_x = (img.width - text_w) // 2
        text_y = img.height - bar_height + (bar_height - text_h) // 2 - 30
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
        return img

    orig_img = add_label(orig_img, "Original")
    new_img = add_label(new_img, "Wavelet Attention (Ours)")
    # Concatenate side by side
    combined = Image.new("RGB", (orig_img.width + new_img.width, orig_img.height))
    combined.paste(orig_img, (0, 0))
    combined.paste(new_img, (orig_img.width, 0))
    # Add image name as a title bar
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default(size=24)
    title_text = f"Image: {image_name}"
    draw.rectangle([0, 0, combined.width, 40], fill=(255, 255, 255, 220))
    draw.text((10, 8), title_text, fill=(0, 0, 0), font=font)
    frames.append(np.array(combined))

# Save as GIF
if frames:
    imageio.mimsave(OUTPUT_GIF, frames, duration=1 * 1000)
    print(f"Animation saved as {OUTPUT_GIF}")
else:
    print("No common overlay images found for animation.")
