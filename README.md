# Biomedical Image Analysis Toolkit (Demo)

Small, self-contained demo repository showcasing:
- clean Python project structure
- basic microscopy/biomedical image preprocessing
- simple segmentation (Otsu + morphology)
- per-object morphological & intensity feature extraction
- CSV export + overlay figure generation

> Note: This repo is a public demo (non-confidential) created to illustrate coding style and reproducibility.
> It does not include any proprietary or institution-confidential code/data.

## Quickstart

```bash
pip install -r requirements.txt
python example_pipeline.py --image path/to/your_image.png --outdir outputs
