import os
import imageio
import numpy as np
from glob import glob
from PIL import Image


def make_gif_for_folder(folder, output_gif="animation.gif", duration=1.0):
    """
    Create a GIF animation from attention_maps and segmentation_results in the given folder.
    """
    attention_dir = os.path.join(folder, "attention_maps")
    segmentation_dir = os.path.join(folder, "segmentation_results")

    # Collect all PNGs in order (sorted by filename)
    attention_imgs = sorted(glob(os.path.join(attention_dir, "*.png")))
    segmentation_imgs = sorted(glob(os.path.join(segmentation_dir, "*.png")))

    # Combine all steps (attention first, then segmentation)
    all_imgs = attention_imgs + segmentation_imgs

    # Read images and resize to the first image's size
    frames = []
    if all_imgs:
        # Use the first image's size as the target
        first_img = Image.open(all_imgs[0])
        target_size = first_img.size  # (width, height)
        for img_path in all_imgs:
            img = (
                Image.open(img_path).convert("RGB").resize(target_size, Image.BILINEAR)
            )
            frames.append(np.array(img))
    else:
        print("No images found to create GIF.")
        return

    # Save as GIF
    output_path = os.path.join(folder, output_gif)
    imageio.mimsave(output_path, frames, duration=int(duration * 1000))
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the image folder (e.g., all_visualizations/daeformer/<image_basename>/)",
    )
    parser.add_argument(
        "--output_gif", type=str, default="animation.gif", help="Output GIF filename"
    )
    parser.add_argument(
        "--duration", type=float, default=1.0, help="Duration per frame in seconds"
    )
    args = parser.parse_args()

    make_gif_for_folder(args.folder, args.output_gif, args.duration)
