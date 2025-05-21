import matplotlib.pyplot as plt

# Data for epoch 399
classes = range(1, 9)
mean_dice = [
    0.881163,
    0.713932,
    0.861293,
    0.817727,
    0.945661,
    0.665711,
    0.911264,
    0.816260,
]
mean_hd95 = [
    7.523249,
    28.265476,
    50.080927,
    40.075049,
    24.253025,
    9.272545,
    40.909898,
    13.960817,
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("DAEFormer Test Results: Epoch 399", fontsize=16)

bars1 = ax1.bar(classes, mean_dice, color="salmon")
ax1.set_xlabel("Class")
ax1.set_ylabel("Mean Dice Score")
ax1.set_title("Epoch 399 - Mean Dice Scores by Class")
ax1.set_xticks(classes)
ax1.set_ylim(0, 1)
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

bars2 = ax2.bar(classes, mean_hd95, color="lightgreen")
ax2.set_xlabel("Class")
ax2.set_ylabel("Mean HD95 Score")
ax2.set_title("Epoch 399 - Mean HD95 Scores by Class")
ax2.set_xticks(classes)
for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
    )

overall_dice = 0.826626
overall_hd95 = 26.792623
fig.text(
    0.5,
    0.02,
    f"Overall Performance - Mean Dice: {overall_dice:.4f}, Mean HD95: {overall_hd95:.4f}",
    ha="center",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.subplots_adjust(bottom=0.18)
plt.savefig("test_results_399.png", dpi=300, bbox_inches="tight")
plt.close()
