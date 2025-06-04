import matplotlib.pyplot as plt
import numpy as np

# Historical training data
epochs = list(range(1, 21))  # 1 to 20
losses = [
    0.4507,
    0.3393,
    0.2994,
    0.2839,
    0.5157,
    0.3572,
    0.3470,
    0.2002,
    0.2510,
    0.1978,
    0.2151,
    0.2538,
    0.3007,
    0.3969,
    0.2284,
    0.1931,
    0.0719,
    0.2023,
    0.1539,
    0.1347,
]
avg_losses = [
    0.4046,
    0.3534,
    0.3358,
    0.3227,
    0.3065,
    0.2960,
    0.2848,
    0.2711,
    0.2651,
    0.2424,
    0.2299,
    0.2224,
    0.2021,
    0.1995,
    0.1739,
    0.1533,
    0.1813,
    0.1358,
    0.1247,
    0.1193,
]


def plot_training_history(epochs, losses, avg_losses, save_path="training_history.png"):
    """Plot the training history with both current and average losses."""
    plt.figure(figsize=(12, 6))

    plt.plot(epochs, avg_losses, "r-o", label="Average Loss", alpha=0.7, markersize=4)

    # Customize the plot
    plt.title("Training Loss History", fontsize=14, pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=10)

    # Set x-axis ticks to show all epochs
    plt.xticks(epochs)

    # Add some padding to y-axis
    plt.margins(y=0.1)

    # Add annotations for min and max points
    min_avg_idx = np.argmin(avg_losses)


    plt.annotate(
        f"Min Avg Loss: {avg_losses[min_avg_idx]:.4f}",
        xy=(epochs[min_avg_idx], avg_losses[min_avg_idx]),
        xytext=(epochs[min_avg_idx] + 1, avg_losses[min_avg_idx] - 0.1),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
    )

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Plot the training history
    plot_training_history(epochs, losses, avg_losses)
    print("Training history plot has been saved as 'training_history.png'")
