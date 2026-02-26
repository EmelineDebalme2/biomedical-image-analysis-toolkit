from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, disk, opening, closing
from skimage.measure import label


def segment_objects(
    img: np.ndarray,
    min_size: int = 200,
    hole_size: int = 200,
    morph_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment foreground objects with Otsu thresholding + morphology.

    Parameters
    ----------
    img : np.ndarray
        2D float image in [0, 1].
    min_size : int
        Minimum object size in pixels.
    hole_size : int
        Fill holes smaller than this.
    morph_radius : int
        Radius for opening/closing.

    Returns
    -------
    mask : np.ndarray
        Boolean mask of segmented objects.
    labels : np.ndarray
        Connected components labels (0 = background).
    """
    if img.ndim != 2:
        raise ValueError("segment_objects expects a 2D grayscale image.")

    t = threshold_otsu(img)
    mask = img > t

    if morph_radius and morph_radius > 0:
        se = disk(morph_radius)
        mask = opening(mask, se)
        mask = closing(mask, se)

    if hole_size and hole_size > 0:
        mask = remove_small_holes(mask, area_threshold=hole_size)

    if min_size and min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)

    labels = label(mask)
    return mask.astype(bool), labels
