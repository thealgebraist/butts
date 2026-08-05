# butts — litter detection dataset for autonomous drone + buggy retrieval

Training and evaluation data for a vision model that detects litter outdoors,
**primarily cigarette butts**, secondarily all other common trash, under the full
range of Danish outdoor conditions — summer glare, winter, rain, damp, snowmelt,
and darkness under artificial light.

## Goal

Build the most precise detector achievable **given the actual optics** of two
platforms:

1. **Drone (DJI Neo 2)** — surveys an area from above, then descends for a
   detailed low pass.
2. **Buggy** — a small ground vehicle with a camera and a simple arm that drives
   to each detection and picks the object up.

Precision matters more than raw recall: a false positive sends the buggy on a
wasted trip, and the arm must be aimed at something that is really there.

### Operating pipeline

| Stage | Platform | Altitude | Role |
|---|---|---|---|
| Survey | Drone | 10–12 m | Large items; terrain/surface map; picks priority regions |
| Detection | Drone | 1.5–2 m | Primary small-object detection, including cigarette butts |
| Approach | Buggy | 0.2–1 m | Identification, grasp planning, retrieval |

## The optical constraint that shapes everything

The Neo 2 has a **16.5 mm-equivalent lens** (very wide) and a 4K sensor. Ground
sample distance therefore degrades quickly with altitude, and a cigarette butt is
only about 8 × 25 mm:

| Altitude | Swath | GSD | Cigarette butt | Bottle |
|---|---|---|---|---|
| 20 m | 43.6 m | 11.4 mm/px | **2.2 px** | **22 px** |
| 10 m | 21.8 m | 5.7 mm/px | **4.4 px** | 44 px |
| 5 m | 10.9 m | 2.8 mm/px | **8.8 px** | 88 px |
| 2 m | 4.4 m | 1.14 mm/px | 22 px | 220 px |
| 1 m | 2.2 m | 0.57 mm/px | 44 px | 440 px |

Detectors need roughly 20 px on an object's longest dimension. So:

- **Cigarette butts are only detectable below ~2.2 m** (~1.6 m in the
  pessimistic FOV reading). This is physics, not a model-quality problem.
- **Even bottles are marginal at 20 m.** The survey pass belongs at 10–12 m.
- Every image in the training set must be labelled with the altitude it was
  taken from, because scale is the dominant variable.

Full derivation, capture plan and offline test protocol:
**[analysis/dataset_plan.pdf](analysis/dataset_plan.pdf)**.

## Current state

| | |
|---|---|
| Images | 244 |
| Capture device | iPhone SE (3rd gen), handheld ≈1.2 m |
| Resolution | 4032 × 3024 |
| Aerial images | **0** |
| Night / rain / snow / damp images | **0** |
| Manually annotated files | 100 (331 polygons, 126 of them objects) |
| Classes | 23 + `dontknow` |

The set proves what the targets look like on the ground in summer daylight. It
does not yet support training for any deployment condition. Open issues track
each gap.

## Conditions the dataset must cover

Every cell below needs images of the primary class (cigarette butts) and, where
practical, the secondary classes. See the issue tracker for per-area targets.

### Season and weather

- **Summer, maximum sunlight** — harsh shadows, blown highlights, high contrast
- **Summer, overcast** — flat diffuse light
- **Bright sky / backlit** — sun low and in frame, lens flare
- **Rain, active** — droplets on lens, rain streaks, moving water
- **After rain** — wet surfaces, specular reflections, puddles, dark backgrounds
- **Damp / ~90 % humidity** — fog, mist, dew, lens condensation
- **Winter, bare ground** — frozen soil, dead vegetation, low sun angle
- **Snow, fresh** — high-key white background, extreme exposure challenge
- **Snow, melting / patchy** — mixed snow and ground, slush, high-contrast edges
- **After snowmelt** — saturated ground, matted flattened vegetation

### Light level

- Daylight (all of the above)
- Dusk / low light
- **Night with omnidirectional LED** — drone and buggy
- **Night with directed LED** — hard shadows, hotspot falloff, specular glare
  off wet or plastic surfaces

### Altitude and viewpoint

| Tier | Height | Angle |
|---|---|---|
| Buggy, close | 0.2 m | 0–30° oblique |
| Buggy, search | 1 m | nadir and 45° |
| Drone, low | 2 m | nadir |
| Drone, transition | 3–5 m | nadir and 30° |
| Drone, survey | 10–12 m | nadir |

The most valuable capture method is the **altitude ladder**: photograph an
identical, untouched scene at 0.2, 1, 2, 3, 5, 10 and 12 m. This gives exact
cross-scale correspondence and measures where each class stops being detectable.

### Surface

Present: gravel, soil, grass.
Missing: asphalt, concrete/paving, sand, mown lawn, leaf litter, wood chip,
snow, and wet variants of all of them.

### Occlusion

Four levels — none, light (<25 %), partial (25–60 %), heavy (>60 %) — against
grass, cut grass, dead leaves, loose soil, gravel, twigs, puddles, and **other
litter**.

### Negatives

Empty ground plus deliberate confusers: pine cones, stones, bark, twigs, leaves,
mushrooms, dog waste. Target 20–30 % of the final set.

## Repository layout

```
images/
  heic/<class>/        original HEIC + manual *_annot.json polygon annotations
  jpg/<class>/         JPEG 90% conversions, same tree
  aerial/ close_up/ lighting/ partial_occlusion/ surfaces/ weather/
                       condition scaffold — currently empty, see issues
analysis/
  dataset_plan.pdf     gap analysis, optics, capture and test-video plan
  dataset_plan.tex     source
  image_analysis.json  per-image vision-model labels
  crops/               polygon-masked object cutouts
scripts/
  analyze_images.py    OpenAI Batch API vision labelling
  reclassify.py        reorganize images by predicted content
  extract_polygons.py  cut annotated objects out via their polygon masks
tools/annotator/       manual polygon annotation tool
```

Images are stored with Git LFS.

## Annotation format

Each image may have a sibling `NAME.heic_annot.json` containing manual polygon
annotations in full-resolution image coordinates:

```json
[{ "label": "cigarette_butt",
   "bbox": {"x": 1517.7, "y": 625.1, "width": 998.2, "height": 1465.4},
   "polygon": [{"x": 1950.8, "y": 721.0}] }]
```

Polygons (not just boxes) are required: the buggy's arm needs an object outline
for grasp planning.

**Known issues with existing labels**: mixed Danish/English (`serviet`,
`grankogler`), truncations (`dirty_clot`, `bottle_`, `gra`), and 205 of 331
polygons label terrain rather than objects. A controlled vocabulary is being
introduced.

## Contributing images

When capturing, record for every image: **altitude, camera angle, surface,
lighting/weather, occlusion level**, and whether artificial light was used.
Without this metadata the evaluation cannot be broken down by condition, and it
will not be possible to tell *where* the model fails.

Include a scale reference of known size in frame where possible — it verifies
GSD and recovers real object dimensions for the arm.
