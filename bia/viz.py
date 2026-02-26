from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb


def save_overlay_figure(
    img: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    title: str = "Segmentation overlay",
) -> None:
    """
    Save an overlay figure (image + colored labels).

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image in [0,1].
    labels : np.ndarray
        2D labeled mask.
    out_path : str | Path
        Output path (png recommended).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = label2rgb(labels, image=img, bg_label=0, alpha=0.35)

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Input")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(overlay)
    ax2.set_title(title)
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
