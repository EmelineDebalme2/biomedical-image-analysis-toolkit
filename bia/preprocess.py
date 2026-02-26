from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from skimage.exposure import rescale_intensity


def preprocess_image(
    img: np.ndarray,
    median_size: int = 3,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Simple preprocessing: median denoising + robust contrast normalization.

    Parameters
    ----------
    img : np.ndarray
        2D float image in [0, 1] (recommended).
    median_size : int
        Median filter size (odd integer recommended).
    p_low, p_high : float
        Percentiles for contrast stretching.

    Returns
    -------
    np.ndarray
        Preprocessed image in [0, 1].
    """
    if img.ndim != 2:
        raise ValueError("preprocess_image expects a 2D grayscale image.")
    if median_size and median_size > 1:
        img = median_filter(img, size=median_size)
    lo, hi = np.percentile(img, [p_low, p_high])
    img = rescale_intensity(img, in_range=(lo, hi), out_range=(0.0, 1.0))
    return img.astype(np.float32)
