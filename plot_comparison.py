import matplotlib.pyplot as plt

# Data from both test results
classes = range(1, 9)  # Classes 1-8

# Final results (Epoch 399)
final_dice = [
    0.881163,
    0.713932,
    0.861293,
    0.817727,
    0.945661,
    0.665711,
    0.911264,
    0.816260,
]
final_hd95 = [
    7.523249,
    28.265476,
    50.080927,
    40.075049,
    24.253025,
    9.272545,
    40.909898,
    13.960817,
]

# Epoch 20 results
epoch20_dice = [
    0.690296,
    0.523122,
    0.645916,
    0.630192,
    0.901557,
    0.298858,
    0.799985,
    0.585676,
]
epoch20_hd95 = [
    38.805548,
    79.531224,
    43.771345,
    46.322048,
    21.693259,
    51.580104,
    36.173331,
    32.853784,
]

# Create figure with four subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("DAEFormer Performance Comparison: Epoch 399 vs Epoch 20", fontsize=16)

# Plot Epoch 399 Dice Scores
bars1 = ax1.bar(classes, final_dice, color="salmon")
ax1.set_xlabel("Class")
ax1.set_ylabel("Mean Dice Score")
ax1.set_title("Epoch 399 - Mean Dice Scores")
ax1.set_xticks(classes)
ax1.set_ylim(0, 1)

# Add value labels on top of bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

# Plot Epoch 399 HD95 Scores
bars2 = ax2.bar(classes, final_hd95, color="lightgreen")
ax2.set_xlabel("Class")
ax2.set_ylabel("Mean HD95 Score")
ax2.set_title("Epoch 399 - Mean HD95 Scores")
ax2.set_xticks(classes)

# Add value labels on top of bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

# Plot Epoch 20 Dice Scores
bars3 = ax3.bar(classes, epoch20_dice, color="salmon")
ax3.set_xlabel("Class")
ax3.set_ylabel("Mean Dice Score")
ax3.set_title("Epoch 20 - Mean Dice Scores")
ax3.set_xticks(classes)
ax3.set_ylim(0, 1)

# Add value labels on top of bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

# Plot Epoch 20 HD95 Scores
bars4 = ax4.bar(classes, epoch20_hd95, color="lightgreen")
ax4.set_xlabel("Class")
ax4.set_ylabel("Mean HD95 Score")
ax4.set_title("Epoch 20 - Mean HD95 Scores")
ax4.set_xticks(classes)

# Add value labels on top of bars
for bar in bars4:
    height = bar.get_height()
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

# Add overall performance as text
final_overall_dice = 0.826626
final_overall_hd95 = 26.792623
epoch20_overall_dice = 0.634450
epoch20_overall_hd95 = 43.841330

fig.text(
    0.5,
    0.02,
    f"Overall Performance:\n"
    f"Epoch 399 - Mean Dice: {final_overall_dice:.4f}, Mean HD95: {final_overall_hd95:.4f}\n"
    f"Epoch 20 - Mean Dice: {epoch20_overall_dice:.4f}, Mean HD95: {epoch20_overall_hd95:.4f}",
    ha="center",
    fontsize=12,
)

# Adjust layout to prevent overlap and leave space for bottom text
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.subplots_adjust(bottom=0.18)

# Save the plot
plt.savefig("performance_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
