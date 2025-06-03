import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from networks.DAEFormer import DAEFormer as DAEFormerOld
from networks.DAEFormer_new import DAEFormer as DAEFormerNew
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import time
import psutil
import os


class ModelEvaluator:
    def __init__(self, num_classes=9, model_path="model_out/best_model.pth"):
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize metrics
        self.metrics = {
            "iou": MulticlassJaccardIndex(num_classes=num_classes).to(self.device),
            "precision": MulticlassPrecision(
                num_classes=num_classes, average="macro"
            ).to(self.device),
            "recall": MulticlassRecall(num_classes=num_classes, average="macro").to(
                self.device
            ),
            "f1": MulticlassF1Score(num_classes=num_classes, average="macro").to(
                self.device
            ),
        }

        # Initialize models
        self.model_old = DAEFormerOld(num_classes=num_classes).to(self.device)
        self.model_new = DAEFormerNew(num_classes=num_classes).to(self.device)

        # Load trained weights
        if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle both direct state dict and dictionary-wrapped state dict
            state_dict = (
                checkpoint
                if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint
                else checkpoint.get("model_state_dict", checkpoint)
            )

            # Try to load weights for both models
            try:
                self.model_old.load_state_dict(state_dict)
                print("Successfully loaded weights for old model")
            except Exception as e:
                print(f"Could not load weights for old model: {e}")

            try:
                self.model_new.load_state_dict(state_dict)
                print("Successfully loaded weights for new model")
            except Exception as e:
                print(f"Could not load weights for new model: {e}")
        else:
            print(
                f"Warning: No model weights found at {model_path}. Using randomly initialized models."
            )

        self.model_old.eval()
        self.model_new.eval()

    def measure_memory_usage(self, model, input_tensor):
        """Measure memory usage during inference"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run inference
        with torch.no_grad():
            _ = model(input_tensor)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        return final_memory - initial_memory

    def measure_inference_time(self, model, input_tensor, num_runs=100):
        """Measure average inference time"""
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                end_time = time.time()
                times.append(end_time - start_time)
        return np.mean(times), np.std(times)

    def calculate_metrics(self, predictions, targets):
        """Calculate all metrics for the predictions"""
        # Ensure predictions are in the correct format (B, H, W) with class indices
        if predictions.dim() == 4:  # If predictions are (B, C, H, W)
            predictions = predictions.argmax(dim=1)  # Convert to class indices

        # Ensure targets are in the correct format (B, H, W)
        if targets.dim() == 4:  # If targets are (B, C, H, W)
            targets = targets.argmax(dim=1)

        # Move tensors to device
        predictions = predictions.to(self.device)
        targets = targets.to(self.device)

        results = {}
        for metric_name, metric in self.metrics.items():
            results[metric_name] = metric(predictions, targets).item()
        return results

    def evaluate_model(self, model, input_tensor, target_tensor):
        """Evaluate a single model"""
        with torch.no_grad():
            # Get predictions
            predictions = model(input_tensor)

            # Calculate metrics
            metrics = self.calculate_metrics(predictions, target_tensor)

            # Measure performance
            memory_usage = self.measure_memory_usage(model, input_tensor)
            inference_time, inference_std = self.measure_inference_time(
                model, input_tensor
            )

            return {
                "metrics": metrics,
                "memory_usage_mb": memory_usage,
                "inference_time_ms": inference_time * 1000,  # Convert to ms
                "inference_std_ms": inference_std * 1000,
            }

    def compare_models(self, input_tensor, target_tensor):
        """Compare both models"""
        print("Evaluating old model (Efficient Attention)...")
        old_results = self.evaluate_model(self.model_old, input_tensor, target_tensor)

        print("Evaluating new model (Wavelet Attention)...")
        new_results = self.evaluate_model(self.model_new, input_tensor, target_tensor)

        # Print comparison
        print("\nModel Comparison Results:")
        print("-" * 50)

        # Compare metrics
        print("\nMetrics Comparison:")
        print(f"{'Metric':<15} {'Old Model':<15} {'New Model':<15} {'Improvement':<15}")
        print("-" * 60)
        for metric in old_results["metrics"].keys():
            old_value = old_results["metrics"][metric]
            new_value = new_results["metrics"][metric]
            improvement = ((new_value - old_value) / old_value) * 100
            print(
                f"{metric:<15} {old_value:<15.4f} {new_value:<15.4f} {improvement:>+15.2f}%"
            )

        # Compare performance
        print("\nPerformance Comparison:")
        print(f"{'Metric':<20} {'Old Model':<15} {'New Model':<15}")
        print("-" * 50)
        print(
            f"{'Memory Usage (MB)':<20} {old_results['memory_usage_mb']:<15.2f} {new_results['memory_usage_mb']:<15.2f}"
        )
        print(
            f"{'Inference Time (ms)':<20} {old_results['inference_time_ms']:<15.2f} {new_results['inference_time_ms']:<15.2f}"
        )
        print(
            f"{'Inference Std (ms)':<20} {old_results['inference_std_ms']:<15.2f} {new_results['inference_std_ms']:<15.2f}"
        )

        return old_results, new_results


def load_kvasir_data(image_path, mask_path):
    """Load image and mask from Kvasir-SEG dataset"""
    # Load and preprocess image
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Load mask
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask = mask.resize((224, 224), Image.NEAREST)  # Use nearest neighbor for masks
    mask_array = np.array(mask)

    # Normalize mask values to range [0, num_classes-1]
    unique_values = np.unique(mask_array)
    num_classes = 9  # Expected number of classes

    # Create a mapping from original values to normalized values
    value_mapping = {val: i for i, val in enumerate(sorted(unique_values))}
    if len(value_mapping) > num_classes:
        # If we have more unique values than classes, map to closest class
        step = len(value_mapping) / num_classes
        value_mapping = {
            val: min(int(i / step), num_classes - 1)
            for i, val in enumerate(sorted(unique_values))
        }

    # Apply the mapping
    normalized_mask = np.vectorize(value_mapping.get)(mask_array)
    mask_tensor = torch.from_numpy(normalized_mask).long()
    mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension

    return image_tensor, mask_tensor


def main():
    # Paths to Kvasir-SEG dataset
    images_dir = os.path.join("datasets", "Kvasir-SEG", "images")
    masks_dir = os.path.join("datasets", "Kvasir-SEG", "masks")

    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Evaluate on each image
    for image_file in image_files:
        print(f"\nEvaluating {image_file}...")

        # Construct paths
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, image_file.replace(".jpg", ".jpg"))

        # Load data
        image_tensor, mask_tensor = load_kvasir_data(image_path, mask_path)

        # Compare models
        evaluator.compare_models(image_tensor, mask_tensor)


if __name__ == "__main__":
    main()
