# Annotations

This directory holds annotation files for every image in the sibling `images/` directory. The folder structure mirrors `images/` exactly.

## Supported Formats

- **YOLO** (`.txt`): one file per image, each line `<class_id> <x_center> <y_center> <width> <height>` (all normalised 0–1).
- **COCO JSON** (`.json`): one file per subcategory, containing bounding boxes and optional segmentation polygons.

## Class IDs

| ID | Label             |
|----|-------------------|
| 0  | cigarette_butt    |

Additional classes may be added later (e.g. whole cigarette, cigarette pack).

## Directory Structure

```
annotations/
├── aerial/
├── close_up/
├── partial_occlusion/
├── lighting/
├── surfaces/
└── weather/
```
