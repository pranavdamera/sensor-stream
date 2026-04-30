# Vision Data — Paired Satellite Change Detection Images

This directory holds before/after/mask image triples for training and evaluating
the PyTorch change-detection model.

## Expected Directory Structure

```
data/vision/
  demo_or_dataset/
    train/
      before/   *.png   — pre-event satellite images
      after/    *.png   — post-event satellite images
      mask/     *.png   — binary change mask (255 = changed, 0 = unchanged)
    val/
      before/
      after/
      mask/
```

Files are matched by stem (filename without extension).  All three directories
must contain a file with the same stem for a sample to be included.

## Suitable Public Datasets

| Dataset | Description | License |
|---------|-------------|---------|
| **LEVIR-CD** | ~637 pairs of 1024×1024 Google Earth images, building change detection | CC BY 4.0 |
| **OSCD** (Onera Satellite Change Detection) | Sentinel-2 multi-spectral pairs, 24 city pairs | CC BY-NC-SA |
| **xView2** | Pre/post disaster building damage, 850k+ annotations | CC BY-NC 4.0 |
| **SpaceNet 7** | Planet monthly mosaics, building footprint change | CC BY-SA 4.0 |

## Quick Demo with a Manual Image Pair

If you only want to test the pipeline end-to-end without a full dataset, prepare
a single pair of cropped satellite images (e.g. from Google Earth time-lapse or
Copernicus Browser) and a hand-drawn binary mask:

```bash
mkdir -p data/vision/demo_or_dataset/train/{before,after,mask}
mkdir -p data/vision/demo_or_dataset/val/{before,after,mask}

# Place your images:
# data/vision/demo_or_dataset/train/before/scene_001.png
# data/vision/demo_or_dataset/train/after/scene_001.png
# data/vision/demo_or_dataset/train/mask/scene_001.png  (binary, 0/255)
```

Then run:

```bash
python scripts/train_change_detector.py --epochs 5 --batch-size 1
python scripts/demo_change_inference.py \
    --before data/vision/demo_or_dataset/val/before/scene_001.png \
    --after  data/vision/demo_or_dataset/val/after/scene_001.png
```

## Downloading LEVIR-CD

```bash
# Official source (requires registration):
# https://justchenhao.github.io/LEVIR/
# After download, reorganize to match the expected layout above.
```

Do not commit large image datasets to this repository.  Add image directories
to `.gitignore` and document the download steps instead.
