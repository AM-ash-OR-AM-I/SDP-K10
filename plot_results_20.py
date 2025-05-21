import matplotlib.pyplot as plt

# Data for epoch 20
classes = range(1, 9)
mean_dice = [
    0.690296,
    0.523122,
    0.645916,
    0.630192,
    0.901557,
    0.298858,
    0.799985,
    0.585676,
]
mean_hd95 = [
    38.805548,
    79.531224,
    43.771345,
    46.322048,
    21.693259,
    51.580104,
    36.173331,
    32.853784,
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("DAEFormer Test Results: Epoch 20", fontsize=16)

bars1 = ax1.bar(classes, mean_dice, color="salmon")
ax1.set_xlabel("Class")
ax1.set_ylabel("Mean Dice Score")
ax1.set_title("Epoch 20 - Mean Dice Scores by Class")
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
ax2.set_title("Epoch 20 - Mean HD95 Scores by Class")
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

overall_dice = 0.634450
overall_hd95 = 43.841330
fig.text(
    0.5,
    0.02,
    f"Overall Performance - Mean Dice: {overall_dice:.4f}, Mean HD95: {overall_hd95:.4f}",
    ha="center",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.subplots_adjust(bottom=0.18)
plt.savefig("test_results_20.png", dpi=300, bbox_inches="tight")
plt.close()
