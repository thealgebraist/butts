# Trash/Waste Image Dataset

A comprehensive image dataset organized by material type for computer vision and object detection training.

## Directory Structure

```
├── datasets/
│   └── waste_items/              # Main image collections organized by material
│       ├── packaging/            # Food and beverage packaging (7 categories)
│       ├── plastics/             # Plastic waste items (8 categories)
│       ├── paper/                # Paper-based materials (4 categories)
│       ├── organic_textiles/     # Natural and textile items (4 categories)
│       ├── metals/               # Metal waste items
│       └── miscellaneous/        # Other miscellaneous items (7 categories)
├── tools/
│   └── annotator/                # Image annotation tool
├── analysis/                     # Analysis results and experiments
│   ├── gray_code_patterns/
│   ├── synthetic_controls/
│   └── work_dir/
├── reference/
│   ├── papers/                   # Research papers and technical documentation
│   └── docs/                     # Additional documentation
└── README.md                      # This file
```

## Dataset Categories

### Packaging (7 categories)
- Burgerking packaging
- Candy wrapper
- Cocio packaging
- Gum package
- McDonalds packaging
- Soda can
- Snus packaging

### Plastics (8 categories)
- Generic plastic items
- Plastic bags
- Plastic bottle pieces
- Plastic boxes
- Plastic filaments
- Plastic pieces
- Plastic wire
- Plastic wrappers

### Paper (4 categories)
- Paper items
- Paper bags
- Paper cups
- Napkins

### Organic & Textiles (4 categories)
- Dirty cloth
- Gloves
- Insect beetle
- Pebbles

### Metals (1 category)
- Rusty metal pieces

### Miscellaneous (7 categories)
- Chairs
- Cigarette butts
- Krukker items
- Grankogler items
- Mystery box contents
- Thrashbags
- Unknown pieces

## File Format

Each dataset category contains:
- Image files (HEIC format)
- Annotation JSON files (format: `{image_name}.heic_annot.json`)

## Tools

- **Annotator**: Image annotation tool with ImGui interface for labeling and analyzing images

## Analysis

- **Gray Code Patterns**: Structured light metrology patterns
- **Synthetic Controls**: Control datasets for validation
- **Work Directory**: Experimental results and processing outputs

## References

See `reference/papers/` for technical documentation including:
- Metrology analysis reports
- Stereo vision studies
- IMU experiment summaries
- Image processing research

---

Dataset organized for machine learning and computer vision applications.
