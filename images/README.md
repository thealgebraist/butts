# Dataset Images

This directory contains raw images of cigarette butts collected "in the wild" for training a neural network detector. Images are organised into subcategories covering the range of conditions the model must handle — from high-altitude drone footage to close-up ground-level shots, across different surfaces, lighting conditions, and levels of occlusion.

## Directory Structure

```
images/
├── aerial/
│   ├── high_altitude/   # Drone imagery from ~10–50 m altitude
│   └── low_altitude/    # Drone imagery from ~1–10 m altitude
│
├── close_up/
│   ├── single_butt/     # One clearly visible butt, unobstructed
│   └── multiple_butts/  # Several butts in a single frame
│
├── partial_occlusion/
│   ├── grass/           # Butt partially hidden under or in grass
│   ├── leaves/          # Butt partially hidden under fallen leaves
│   ├── soil/            # Butt embedded in or covered by dirt/soil
│   └── debris/          # Butt hidden by general litter or debris
│
├── lighting/
│   ├── bright_sunlight/ # Harsh direct sunlight, strong shadows
│   ├── overcast/        # Diffuse, even lighting
│   ├── shadow/          # Butt located in shade or cast shadow
│   └── night/           # Low-light or artificial-light conditions
│
├── surfaces/
│   ├── pavement/        # Concrete, asphalt, tarmac
│   ├── grass/           # Short or long grass backgrounds
│   ├── gravel/          # Loose gravel or stone paths
│   ├── sand/            # Sandy ground (beach, playground, etc.)
│   ├── soil/            # Bare earth or muddy ground
│   └── wet/             # Any wet surface (rain-slicked pavement, mud)
│
└── weather/
    ├── dry/             # Clear, dry conditions
    ├── rain/            # During or just after rainfall
    └── snow/            # Snow-covered ground with visible butts
```

## Naming Convention

Images should be named using the following pattern:

```
<category>_<subcategory>_<YYYYMMDD>_<sequence>.jpg
```

Example: `aerial_high_altitude_20240315_001.jpg`

## Annotation Format

Corresponding annotation files (bounding boxes, segmentation masks) live in the sibling `annotations/` directory, mirroring this structure. Use YOLO `.txt` format or COCO JSON as appropriate.
