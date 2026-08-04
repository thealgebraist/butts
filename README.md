# butts

A dataset of cigarette butts photographed "in the wild", intended for training a neural network to detect them across a wide range of conditions — from drone altitude down to ground level, on various surfaces, under different lighting and weather, and when partially obscured by grass or other debris.

The end goal is to power an autonomous detection-and-retrieval system: a drone identifies butts from the air, and a small ground vehicle navigates to them and picks them up.

## Repository Structure

```
butts/
├── images/          # Raw images, organised by capture scenario
│   ├── aerial/          # High- and low-altitude drone shots
│   ├── close_up/        # Ground-level macro shots
│   ├── partial_occlusion/  # Butts hidden by grass, leaves, soil, debris
│   ├── lighting/        # Varying light conditions (sun, shade, night, …)
│   ├── surfaces/        # Different ground types (pavement, grass, gravel, …)
│   └── weather/         # Dry, rain, and snow conditions
│
└── annotations/     # Bounding-box / segmentation labels mirroring images/
```

See [`images/README.md`](images/README.md) for full folder descriptions and the naming convention, and [`annotations/README.md`](annotations/README.md) for label formats.

## Contributing Images

1. Place images in the most specific matching subcategory folder.
2. Follow the naming convention: `<category>_<subcategory>_<YYYYMMDD>_<sequence>.jpg`
3. Add a corresponding annotation file in `annotations/` using YOLO `.txt` format.
4. Open a pull request with a brief description of the capture conditions.
