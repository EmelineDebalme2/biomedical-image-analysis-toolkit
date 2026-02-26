from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table


def extract_region_features(
    labels: np.ndarray,
    intensity_image: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Extract morphological + intensity features per connected component.

    Parameters
    ----------
    labels : np.ndarray
        Labeled image (0 = background).
    intensity_image : np.ndarray, optional
        Grayscale image used for intensity statistics.

    Returns
    -------
    pd.DataFrame
        One row per object with features.
    """
    if labels.ndim != 2:
        raise ValueError("extract_region_features expects 2D labels.")
    props = [
        "label",
        "area",
        "perimeter",
        "bbox",
        "centroid",
        "eccentricity",
        "solidity",
        "extent",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "equivalent_diameter",
    ]

    if intensity_image is not None:
        props += ["mean_intensity", "min_intensity", "max_intensity"]

    tbl = regionprops_table(labels, intensity_image=intensity_image, properties=props)
    df = pd.DataFrame(tbl)

    # Derived features (robust, readable, commonly used)
    eps = 1e-12
    df["aspect_ratio"] = (df["major_axis_length"] + eps) / (df["minor_axis_length"] + eps)
    df["circularity"] = (4.0 * np.pi * df["area"]) / ((df["perimeter"] + eps) ** 2)

    # bbox columns expansion
    # skimage gives bbox-0..3
    if {"bbox-0", "bbox-1", "bbox-2", "bbox-3"}.issubset(df.columns):
        df = df.rename(
            columns={
                "bbox-0": "bbox_rmin",
                "bbox-1": "bbox_cmin",
                "bbox-2": "bbox_rmax",
                "bbox-3": "bbox_cmax",
            }
        )

    # centroids
    if {"centroid-0", "centroid-1"}.issubset(df.columns):
        df = df.rename(columns={"centroid-0": "centroid_row", "centroid-1": "centroid_col"})

    # Make column order nicer
    preferred = [
        "label",
        "area",
        "perimeter",
        "equivalent_diameter",
        "circularity",
        "eccentricity",
        "aspect_ratio",
        "solidity",
        "extent",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "centroid_row",
        "centroid_col",
        "bbox_rmin",
        "bbox_cmin",
        "bbox_rmax",
        "bbox_cmax",
        "mean_intensity",
        "min_intensity",
        "max_intensity",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]
