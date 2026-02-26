from __future__ import annotations

from pathlib import Path
import numpy as np
from skimage import io, img_as_float32


def load_grayscale_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk as a float32 grayscale array in [0, 1].

    Supports common microscopy formats readable by scikit-image (png, tif, jpg, ...).

    Parameters
    ----------
    path : str or Path
        Path to an image file.

    Returns
    -------
    np.ndarray
        2D float32 image in [0, 1].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = io.imread(str(path))
    if img.ndim == 3:
        # Convert RGB -> grayscale using luminance weights
        img = img[..., :3]  # safety if alpha present
        img = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])

    img = img_as_float32(img)
    # Ensure range is [0, 1] robustly (some tifs may not be)
    img_min, img_max = float(np.min(img)), float(np.max(img))
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img.astype(np.float32)
