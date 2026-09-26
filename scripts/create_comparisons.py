from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GT_DIR = Path("renders/comparisons/test/gt-rgb")
PRED_DIR = Path("renders/comparisons/test/rgb")
OUTPUT_DIR = Path("renders/comparisons/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_comparison(filename):
    gt_path = GT_DIR / filename
    pred_path = PRED_DIR / filename

    # Matplotlib loads JPEGs directly as RGB arrays.
    gt = plt.imread(gt_path)
    pred = plt.imread(pred_path)

    # Convert to float in the range [0, 1].
    gt = gt.astype(np.float32) / 255.0
    pred = pred.astype(np.float32) / 255.0

    # Absolute difference between corresponding pixels.
    error = np.abs(gt - pred)

    # Amplify error only for visualization.
    error_visual = np.clip(error * 3.0, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(gt)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(pred)
    axes[1].set_title("3D Gaussian Splatting")
    axes[1].axis("off")

    axes[2].imshow(error_visual)
    axes[2].set_title("Absolute Error (3x)")
    axes[2].axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{Path(filename).stem}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


for gt_path in sorted(GT_DIR.glob("*.jpg")):
    create_comparison(gt_path.name)