from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bia.io_utils import load_grayscale_image
from bia.preprocess import preprocess_image
from bia.segmentation import segment_objects
from bia.features import extract_region_features
from bia.viz import save_overlay_figure


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Example biomedical image analysis pipeline: preprocess -> segment -> features -> exports."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image (png/tif/jpg).")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--min-size", type=int, default=200, help="Minimum object size (px).")
    parser.add_argument("--hole-size", type=int, default=200, help="Fill holes smaller than this (px).")
    parser.add_argument("--morph-radius", type=int, default=2, help="Morphology radius for opening/closing.")
    parser.add_argument("--median", type=int, default=3, help="Median filter size.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = load_grayscale_image(args.image)
    img_p = preprocess_image(img, median_size=args.median)

    mask, labels = segment_objects(
        img_p,
        min_size=args.min_size,
        hole_size=args.hole_size,
        morph_radius=args.morph_radius,
    )

    df = extract_region_features(labels, intensity_image=img_p)

    # Exports
    csv_path = outdir / "region_features.csv"
    df.to_csv(csv_path, index=False)

    fig_path = outdir / "overlay.png"
    save_overlay_figure(img_p, labels, fig_path)

    summary = {
        "n_objects": int((df.shape[0])),
        "csv": str(csv_path),
        "overlay": str(fig_path),
    }
    print("Done.")
    print(summary)


if __name__ == "__main__":
    main()
