# Object-Centered Information Transport, Mask Inference, Thumbprints, FFNN Classification, and Gaussian Splatting

_A unified technical note from the discussion._

**Scenario.**  
We want to classify physical objects from HD video recordings. Each object is kept near the center of the frame while the camera moves around it in a 360° path, from far away to close up, for example from 10 m to 0.5 m. The video may be monocular or stereo. The central idea is to use a radially weighted information transport rule: information near the center of the image is preserved strongly, while information near the edge is compressed more aggressively.

The goal is not merely to store the video. The goal is to extract the parts of the video that carry actual object-relevant information, infer object masks without having ground-truth masks, compress the object into a compact thumbprint, and use that thumbprint for classification, for example with a classic 1024-neuron fully connected neural network.

---

## Table of Contents

1. [Core Idea](#1-core-idea)  
2. [Capture Setup](#2-capture-setup)  
3. [Radial Information Transport](#3-radial-information-transport)  
4. [Concrete Information Calculation](#4-concrete-information-calculation)  
5. [Monocular Video vs Stereo Video](#5-monocular-video-vs-stereo-video)  
6. [Deriving Masks Without Ground-Truth Masks](#6-deriving-masks-without-ground-truth-masks)  
7. [Mask Precision Analysis](#7-mask-precision-analysis)  
8. [Using a Classic 1024-Neuron FFNN for 512 Objects](#8-using-a-classic-1024-neuron-ffnn-for-512-objects)  
9. [Extracting a Compressed Object Thumbprint](#9-extracting-a-compressed-object-thumbprint)  
10. [Connection to Gaussian Splatting](#10-connection-to-gaussian-splatting)  
11. [Best Practical System](#11-best-practical-system)  
12. [Failure Modes and Limits](#12-failure-modes-and-limits)  
13. [Implementation Sketch](#13-implementation-sketch)  
14. [Summary](#14-summary)  
15. [References and Related Concepts](#15-references-and-related-concepts)

---

# 1. Core Idea

The basic data collection method is:

\[
\text{object} \rightarrow \text{centered 360° video} \rightarrow \text{radially weighted information} \rightarrow \text{mask inference} \rightarrow \text{compressed object representation}
\]

The key axiom is:

\[
\boxed{\text{Information close to the center is more likely to be object-relevant.}}
\]

There is no known object mask at capture time. The object is simply kept in the center. From that weak assumption, we try to infer a mask statistically by using all frames together.

The central idea is:

\[
\text{center prior} + \text{multi-frame consistency} + \text{appearance statistics} + \text{optional stereo depth}
\]

to infer:

\[
P(\text{pixel belongs to object} \mid \text{all frames})
\]

Then, instead of feeding raw video to a neural network, compress the inferred object evidence into a compact thumbprint.

---

# 2. Capture Setup

A typical object capture sequence:

- HD video: \(1920 \times 1080\)
- RGB: 24 bits per pixel
- 10 seconds
- 30 fps
- 300 frames
- Camera moves around the object 360°
- Camera distance changes from 10 m to 0.5 m
- Object is kept near the image center
- Optional stereo camera gives left/right views and depth/disparity

A discrete version used in earlier calculations:

| Distance | Number of views |
|---:|---:|
| 8 m | 36 |
| 4 m | 36 |
| 2 m | 36 |
| 1 m | 36 |
| 0.5 m | 36 |

Total:

\[
5 \times 36 = 180
\]

monocular frames, or 180 stereo pairs.

For a continuous 10-second recording at 30 fps:

\[
10 \times 30 = 300
\]

frames.

---

# 3. Radial Information Transport

Let the image center be:

\[
c=(c_x,c_y)
\]

For a pixel or patch \(u=(x,y)\), define normalized radius:

\[
r(u)=\frac{\sqrt{(x-c_x)^2+(y-c_y)^2}}{R}
\]

where \(R\) is a reference image radius, for example half the image height:

\[
R=540
\]

for a \(1920 \times 1080\) image.

The radial information retention function is:

\[
w_e(r)
=
e+(1-e)
\frac{e^{-2r^2}-e^{-2}}{1-e^{-2}}
\]

where:

- \(r=0\): image center
- \(r=1\): image edge
- \(e\): retained information fraction at the edge

Thus:

\[
w_e(0)=1
\]

and:

\[
w_e(1)=e
\]

The edge-retention values considered were:

\[
64\%, 32\%, 16\%, 8\%, 4\%, 2\%
\]

or:

\[
e \in \{0.64,0.32,0.16,0.08,0.04,0.02\}
\]

The important interpretation is:

\[
w_e(r) = \text{reliability or bandwidth assigned to evidence at radius } r
\]

It is **not** itself an object mask. A pixel near the edge may still belong to the object, but its transported evidence is weaker.

---

# 4. Concrete Information Calculation

## 4.1 Object Size Model

Assume the object is centered and has approximate projected radii:

| Distance | Object radius in image | Normalized radius |
|---:|---:|---:|
| 8 m | 34 px | 0.063 |
| 4 m | 68 px | 0.126 |
| 2 m | 135 px | 0.250 |
| 1 m | 270 px | 0.500 |
| 0.5 m | 486 px | 0.900 |

The close-up frame at 0.5 m is the hardest for radial compression because the object nearly fills the frame.

---

## 4.2 Average Object Information Kept

Assuming the object fills a disk of normalized radius \(a\), the average retained object information is approximately:

\[
\bar w_e(a)
=
\frac{2}{a^2}
\int_0^a r w_e(r) \, dr
\]

Using the values above:

| Edge kept | 8 m | 4 m | 2 m | 1 m | 0.5 m |
|---:|---:|---:|---:|---:|---:|
| 64% | 99.8% | 99.3% | 97.5% | 91.1% | 79.0% |
| 32% | 99.7% | 98.8% | 95.3% | 83.2% | 60.3% |
| 16% | 99.6% | 98.5% | 94.2% | 79.3% | 51.0% |
| 8% | 99.6% | 98.3% | 93.6% | 77.3% | 46.3% |
| 4% | 99.6% | 98.3% | 93.3% | 76.3% | 43.9% |
| 2% | 99.6% | 98.2% | 93.2% | 75.9% | 42.8% |

Main conclusions:

1. Edge retention barely matters at 8 m, 4 m, and 2 m because the object is still close to the high-information center.
2. Edge retention matters strongly at 1 m and especially 0.5 m.
3. If the object nearly fills the image, very low edge retention damages boundary and shape information.

---

## 4.3 Total Useful Object Information

Using earlier illustrative assumptions, approximate useful object-region information across the multi-distance sequence was:

| Edge kept | Monocular video | Stereo video | Stereo gain |
|---:|---:|---:|---:|
| 64% | 0.751 Gbit | 1.149 Gbit | 1.53× |
| 32% | 0.614 Gbit | 0.933 Gbit | 1.52× |
| 16% | 0.546 Gbit | 0.825 Gbit | 1.51× |
| 8% | 0.512 Gbit | 0.771 Gbit | 1.51× |
| 4% | 0.494 Gbit | 0.744 Gbit | 1.51× |
| 2% | 0.486 Gbit | 0.731 Gbit | 1.50× |

These are not exact Shannon information numbers. They are an illustrative “equivalent useful object evidence” model.

Main conclusion:

\[
\boxed{\text{Stereo gives about 1.5× effective object information, not 2×, because the two views are correlated.}}
\]

---

## 4.4 Close Frames Dominate

Close-up frames carry far more object pixels.

For representative edge values:

| Edge kept | Distance | Monocular contribution | Stereo contribution |
|---:|---:|---:|---:|
| 64% | 8 m | 0.003 Gbit | 0.003 Gbit |
| 64% | 4 m | 0.012 Gbit | 0.014 Gbit |
| 64% | 2 m | 0.048 Gbit | 0.060 Gbit |
| 64% | 1 m | 0.180 Gbit | 0.261 Gbit |
| 64% | 0.5 m | 0.506 Gbit | 0.810 Gbit |
| 16% | 8 m | 0.003 Gbit | 0.003 Gbit |
| 16% | 4 m | 0.012 Gbit | 0.014 Gbit |
| 16% | 2 m | 0.047 Gbit | 0.058 Gbit |
| 16% | 1 m | 0.157 Gbit | 0.228 Gbit |
| 16% | 0.5 m | 0.327 Gbit | 0.523 Gbit |
| 2% | 8 m | 0.003 Gbit | 0.003 Gbit |
| 2% | 4 m | 0.012 Gbit | 0.014 Gbit |
| 2% | 2 m | 0.046 Gbit | 0.058 Gbit |
| 2% | 1 m | 0.150 Gbit | 0.218 Gbit |
| 2% | 0.5 m | 0.274 Gbit | 0.439 Gbit |

Main conclusion:

\[
\boxed{\text{Close-up frames are the richest, but they are most damaged by aggressive edge compression.}}
\]

---

# 5. Monocular Video vs Stereo Video

## 5.1 Monocular Video

Monocular HD video gives:

- color
- texture
- silhouette
- material appearance
- scale changes
- motion parallax from camera movement
- multi-view coverage

It is especially useful for objects where class is determined by:

- color
- printed labels
- surface texture
- silhouette
- local shape

Examples:

- blue rubber glove
- cigarette butt
- plastic wrapper
- soda can
- bottle cap
- paper cup

---

## 5.2 Stereo Video

Stereo video gives two views:

\[
(X_t^L, X_t^R)
\]

The extra information is:

\[
I(Y;X_t^R \mid X_t^L)
\]

where \(Y\) is the object class.

Stereo does not simply double useful classification information because the two images are highly correlated. Its main value is depth.

Stereo helps with:

- object/background separation
- true physical scale
- 3D shape
- thinness/thickness
- foreground segmentation
- occlusion handling
- reconstructing a 3D model
- initializing Gaussian splats

Depth from stereo follows approximately:

\[
d = \frac{fB}{Z}
\]

where:

- \(d\): disparity
- \(f\): focal length
- \(B\): stereo baseline
- \(Z\): distance

Depth uncertainty roughly scales as:

\[
\sigma_Z \approx \frac{Z^2}{fB}\sigma_d
\]

Thus stereo depth is much more accurate close up than far away.

---

## 5.3 Distance Dependence

| Distance | Monocular value | Stereo value |
|---:|---|---|
| 10 m / 8 m | detection and context | weak depth for small objects |
| 4 m | better object visibility | moderate depth |
| 2 m | strong appearance | useful depth |
| 1 m | strong appearance and silhouette | strong depth |
| 0.5 m | fine material detail | strong depth if focus and baseline allow |

Practical conclusion:

\[
\boxed{\text{Stereo helps most from about 2 m inward.}}
\]

---

# 6. Deriving Masks Without Ground-Truth Masks

There is no actual mask. The object is just kept in the center. Therefore the mask is a latent variable.

For frame \(t\), let:

\[
M_t(x,y) \in \{0,1\}
\]

where \(M_t(x,y)=1\) means the pixel belongs to the object.

We estimate the posterior probability:

\[
q_t(x,y)
=
P(M_t(x,y)=1 \mid \text{all frames})
\]

So the output is initially a **soft mask**, not a hard mask.

---

## 6.1 Center Prior

Define:

\[
\pi(r)=P(M=1\mid r)
=
e^{-\frac{r^2}{2\sigma_c^2}}
\]

For example, with \(\sigma_c=0.35\):

| Radius | Prior object probability |
|---:|---:|
| 0.0 | 100% |
| 0.25 | 77% |
| 0.5 | 36% |
| 0.75 | 10% |
| 1.0 | 1.7% |

This is not the final mask. It is merely the initial belief.

---

## 6.2 Posterior Mask Formula

For a pixel or patch \(u\), define the log-odds score:

\[
L_t(u)
=
\log \frac{\pi(r_u)}{1-\pi(r_u)}
+
\lambda_A w_e(r_u) A_t(u)
+
\lambda_C w_e(r_u) C_t(u)
+
\lambda_S S_t(u)
\]

where:

| Term | Meaning |
|---|---|
| \(\pi(r_u)\) | center prior |
| \(w_e(r_u)\) | radial information reliability |
| \(A_t(u)\) | appearance evidence |
| \(C_t(u)\) | multi-frame consistency evidence |
| \(S_t(u)\) | stereo depth evidence |
| \(\lambda_A,\lambda_C,\lambda_S\) | weights |

Then:

\[
q_t(u)=
\frac{1}{1+e^{-L_t(u)}}
\]

The hard mask is:

\[
\hat M_t(u)
=
\begin{cases}
1 & q_t(u)>\tau \\
0 & q_t(u)\le \tau
\end{cases}
\]

Typical thresholds:

| Threshold | Meaning |
|---:|---|
| \(\tau=0.3\) | high recall |
| \(\tau=0.5\) | balanced |
| \(\tau=0.7\) | high precision |
| \(\tau=0.8\) | conservative mask |

---

## 6.3 Appearance Evidence

Let \(F_t(u)\) be a local patch feature. This could be:

- RGB color
- HSV histogram
- local gradients
- texture descriptor
- DCT coefficients
- CNN embedding
- small patch vector

Estimate:

\[
p(F\mid O)
\]

from high-center-prior pixels and:

\[
p(F\mid B)
\]

from low-center-prior pixels.

Then:

\[
A_t(u)
=
\log \frac{p(F_t(u)\mid O)}{p(F_t(u)\mid B)}
\]

If this is positive, the patch looks object-like. If negative, it looks background-like.

---

## 6.4 Multi-Frame Consistency Evidence

A true object patch should remain statistically attached to the center as the camera moves.

Define:

\[
C_t(u)
=
\text{consistency of patch }u\text{ with object-like regions in nearby frames}
\]

A patch is more likely object if it:

- persists across nearby frames
- moves coherently with the centered object
- belongs to a stable connected region
- changes smoothly with viewpoint
- is not a one-frame background artifact

This is the main mechanism that upgrades the center prior into a real object mask.

---

## 6.5 Stereo Evidence

If stereo is available, compute disparity/depth \(D_t(u)\). Then:

\[
S_t(u)
=
\log \frac{p(D_t(u)\mid O)}{p(D_t(u)\mid B)}
\]

Stereo helps when the object is at a different depth from the background.

For example, a glove on grass may be separated by depth even when RGB color and texture are confusing.

---

## 6.6 EM-Style Algorithm

### Step 1: Initialize

\[
q_t^{(0)}(u)=\pi(r_u)
\]

### Step 2: Estimate object/background statistics

Using soft weights:

\[
p(F\mid O)
\]

is estimated mostly from high-\(q\) pixels, and:

\[
p(F\mid B)
\]

from low-\(q\) pixels.

### Step 3: Update masks

\[
q_t^{(k+1)}(u)
=
\sigma
\left(
\log \frac{\pi(r_u)}{1-\pi(r_u)}
+
\lambda_A w_e(r_u) A_t(u)
+
\lambda_C w_e(r_u) C_t(u)
+
\lambda_S S_t(u)
\right)
\]

### Step 4: Smooth spatially and temporally

A possible energy is:

\[
E(M)
=
\sum_{t,u} -\log P(M_t(u))
+
\beta \sum_{u\sim v} [M_t(u)\ne M_t(v)]
+
\gamma \sum_{\text{tracks}} [M_t(u)\ne M_s(v)]
\]

### Step 5: Repeat

Iterate until \(q_t\) stabilizes.

---

# 7. Mask Precision Analysis

The mask precision depends strongly on:

- object size in the frame
- edge retention \(e\)
- number of views
- stereo availability
- object/background contrast
- stability of centering
- blur
- lighting
- texture

A simple model assumes mask error mostly appears as boundary uncertainty.

Let:

\[
\sigma_{\partial M}
=
\frac{64.8}{\sqrt{36 \cdot w_e(a) \cdot s}}
\]

where:

- \(a\): normalized object radius
- \(w_e(a)\): information kept near object boundary
- \(s=1\): monocular
- \(s>1\): stereo evidence multiplier

Illustrative stereo multipliers:

| Distance | Stereo evidence multiplier |
|---:|---:|
| 8 m | 1.05× |
| 4 m | 1.10× |
| 2 m | 1.25× |
| 1 m | 1.45× |
| 0.5 m | 1.60× |

---

## 7.1 Boundary Precision at 0.5 m

At 0.5 m:

\[
a=0.9R=486 \text{ px}
\]

| Edge kept | Boundary info kept | Monocular boundary error | Monocular F1 | Stereo boundary error | Stereo F1 |
|---:|---:|---:|---:|---:|---:|
| 64% | 66.6% | 13.2 px | 0.978 | 10.5 px | 0.983 |
| 32% | 36.9% | 17.8 px | 0.971 | 14.1 px | 0.977 |
| 16% | 22.1% | 23.0 px | 0.962 | 18.2 px | 0.970 |
| 8% | 14.7% | 28.2 px | 0.954 | 22.3 px | 0.963 |
| 4% | 10.9% | 32.6 px | 0.948 | 25.8 px | 0.958 |
| 2% | 9.1% | 35.8 px | 0.943 | 28.3 px | 0.954 |

Main result:

\[
\boxed{\text{At 0.5 m, 2% edge retention gives about 2× worse boundary precision than 32%.}}
\]

---

## 7.2 At 32% Edge Retention Across Distances

| Distance | Object radius | Boundary info kept | Monocular boundary error | Monocular F1 | Stereo boundary error | Stereo F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 m | 34 px | 99.4% | 10.8 px | 0.750 | 10.6 px | 0.756 |
| 4 m | 68 px | 97.5% | 10.9 px | 0.872 | 10.4 px | 0.878 |
| 2 m | 135 px | 90.8% | 11.3 px | 0.933 | 10.1 px | 0.940 |
| 1 m | 270 px | 69.1% | 13.0 px | 0.962 | 10.8 px | 0.968 |
| 0.5 m | 486 px | 36.9% | 17.8 px | 0.971 | 14.1 px | 0.977 |

Far frames are not mask-precise because the object is tiny. Close frames produce better masks even if the boundary error in pixels is larger.

---

## 7.3 Overall Mask Precision

Area-weighted illustrative results:

| Edge kept | Modality | Precision | Recall | F1 |
|---:|---|---:|---:|---:|
| 64% | Monocular | 0.973 | 0.972 | 0.971 |
| 64% | Stereo | 0.978 | 0.976 | 0.976 |
| 32% | Monocular | 0.968 | 0.966 | 0.965 |
| 32% | Stereo | 0.973 | 0.972 | 0.971 |
| 16% | Monocular | 0.962 | 0.960 | 0.958 |
| 16% | Stereo | 0.969 | 0.967 | 0.966 |
| 8% | Monocular | 0.957 | 0.954 | 0.952 |
| 8% | Stereo | 0.964 | 0.962 | 0.961 |
| 4% | Monocular | 0.954 | 0.949 | 0.947 |
| 4% | Stereo | 0.961 | 0.958 | 0.956 |
| 2% | Monocular | 0.952 | 0.945 | 0.944 |
| 2% | Stereo | 0.959 | 0.955 | 0.954 |

These values assume favorable conditions: stable centering, enough views, decent image quality, and object/background separability.

---

## 7.4 Why Center Prior Alone Is Not Enough

Suppose we used only a fixed central circle of radius \(0.5R\). Then:

| Distance | True object radius | Precision of fixed center mask |
|---:|---:|---:|
| 8 m | 34 px | 1.6% |
| 4 m | 68 px | 6.4% |
| 2 m | 135 px | 25% |
| 1 m | 270 px | 100% |
| 0.5 m | 486 px | 100% precision but only 31% recall |

So the center prior by itself is crude. Its power comes only after combining it with multi-frame and appearance evidence.

---

# 8. Using a Classic 1024-Neuron FFNN for 512 Objects

Assume a classic one-hidden-layer fully connected neural network:

\[
\hat y
=
\operatorname{softmax}
\left(
W_2 \phi(W_1 x+b_1)+b_2
\right)
\]

where:

- \(x\): input vector
- hidden layer: 1024 neurons
- output layer: 512 classes

The number of parameters is approximately:

\[
P \approx 1024d + 1024 \cdot 512
\]

where \(d\) is the input dimension.

---

## 8.1 Raw Video Is Impossible

A 10-second HD RGB video at 30 fps has:

\[
1920 \times 1080 \times 3 \times 300
=
1{,}866{,}240{,}000
\]

input values.

A raw-video FFNN would need:

\[
1024 \times 1{,}866{,}240{,}000
\approx
1.91 \times 10^{12}
\]

weights in the first layer.

That is about 1.9 trillion weights.

At 32-bit floats:

\[
1.91 \times 10^{12} \times 4
\approx
7.6 \text{ TB}
\]

just for the first layer.

So:

\[
\boxed{\text{Raw HD video into a 1024-neuron FFNN is computationally and statistically unrealistic.}}
\]

Even a single raw HD frame requires:

\[
1024 \times 6{,}220{,}800
\approx
6.37 \times 10^9
\]

weights.

---

## 8.2 The Label Only Contains 9 Bits

For 512 object classes:

\[
\log_2(512)=9
\]

So the final answer requires only 9 bits of identity information.

The challenge is not lack of information. The challenge is:

\[
\boxed{\text{compress the video while preserving class-relevant information.}}
\]

---

## 8.3 Effective Training Examples

512 videos × 300 frames:

\[
512 \times 300 = 153{,}600
\]

frames.

But consecutive frames are highly correlated. The effective number of meaningfully different views may be closer to:

\[
512 \times 40 = 20{,}480
\]

to:

\[
512 \times 100 = 51{,}200
\]

This is not enough for raw pixels, but it may be enough for compact object features.

---

## 8.4 Parameter Count After Compression

| Input to FFNN | Input dimension \(d\) | Approximate parameters |
|---|---:|---:|
| Raw 10s HD video | 1,866,240,000 | 1.91 trillion |
| One raw HD frame | 6,220,800 | 6.37 billion |
| 256×256 RGB crop | 196,608 | 202 million |
| 128×128 RGB crop | 49,152 | 50.9 million |
| 64×64 RGB crop | 12,288 | 13.1 million |
| 4096-dimensional feature vector | 4096 | 4.72 million |
| 2048-dimensional feature vector | 2048 | 2.62 million |
| 1024-dimensional feature vector | 1024 | 1.57 million |

Thus, a 1024-neuron FFNN becomes plausible only after reducing the video to a compact feature vector.

---

## 8.5 Expected Classification Performance

Assume the FFNN receives compressed object features, not raw video.

| Scenario | Expected performance |
|---|---:|
| Train/test frames from same video | 95–99%, but inflated |
| Held-out view sectors from same recording | 85–97% |
| New recording of same physical objects | 70–90% |
| New instances of same categories, one video per category | 20–60% |
| Raw HD video into FFNN | infeasible |

The distinction is crucial:

- Identifying the same physical object from the same video is easier.
- Learning a general category from one example is much harder.

A plain FFNN has no built-in image invariances. It does not naturally know about translation, rotation, local texture, or view geometry. Therefore it needs the front-end compression/mask/thumbprint system.

---

# 9. Extracting a Compressed Object Thumbprint

A thumbprint is a compact representation:

\[
T(V) \in \mathbb{R}^d
\]

for one video \(V\).

The objective is:

\[
T^*
=
\arg\max_T I(T(V);Y)-\lambda |T|
\]

where:

- \(Y\): class or identity
- \(I(T(V);Y)\): class-relevant information
- \(|T|\): representation size

In plain terms:

\[
\boxed{\text{Keep the bits that identify the object; discard redundant/background bits.}}
\]

---

## 9.1 Raw Video Size vs Thumbprint Size

A 10-second uncompressed HD RGB video has about:

\[
1.87 \text{ GB}
\]

A thumbprint could be:

| Thumbprint | Size |
|---|---:|
| 4096 fp32 values | 16 KB |
| 4096 fp16 values | 8 KB |
| 2048 fp16 values | 4 KB |
| 1024 fp16 values | 2 KB |
| 1024-bit binary hash | 128 bytes |
| 256-bit binary hash | 32 bytes |

So a natural compression chain is:

\[
1.87\text{ GB}
\rightarrow
2\text{–}16\text{ KB}
\]

for a classifier thumbprint.

---

## 9.2 Relevance Field

For each frame \(t\) and patch \(u\), define relevance:

\[
R_t(u)
\in [0,1]
\]

This says how much that patch should contribute to the thumbprint.

A useful formula is:

\[
R_t(u)
=
P_{\text{center}}(u)
\cdot
w_e(r(u))
\cdot
\sigma
\left(
\alpha A_t(u)+\beta C_t(u)
\right)
\]

or with stereo:

\[
R_t(u)
=
P_{\text{center}}(u)
\cdot
w_e(r(u))
\cdot
\sigma
\left(
\alpha A_t(u)+\beta C_t(u)+\gamma S_t(u)
\right)
\]

where:

| Term | Meaning |
|---|---|
| \(P_{\text{center}}\) | center prior |
| \(w_e(r)\) | radial evidence reliability |
| \(A_t\) | appearance likelihood |
| \(C_t\) | multi-frame consistency |
| \(S_t\) | stereo depth evidence |
| \(\sigma\) | logistic squashing |

---

## 9.3 Weighted Frame Features

Let \(\phi(I_t,u)\) be a patch descriptor.

Then the frame-level object feature is:

\[
z_t
=
\frac{
\sum_u R_t(u)\phi(I_t,u)
}{
\sum_u R_t(u)
}
\]

This says:

\[
\boxed{\text{average only the relevant object-like content, not the whole frame.}}
\]

Possible descriptor types:

### Appearance features

- RGB histogram
- HSV histogram
- saturation/brightness
- color proportions

### Texture features

- gradients
- edge density
- wrinkle frequency
- DCT coefficients
- roughness/smoothness

### Shape features

From \(R_t\):

- area
- perimeter
- elongation
- compactness
- number of lobes
- skeleton length
- radial mass distribution
- boundary curvature

### Multi-scale features

Because the camera moves from far to close:

- far scale: detection/context
- medium scale: silhouette
- close scale: material/texture
- very close scale: fine detail

---

## 9.4 Frame Novelty and Selection

Do not use all 300 frames equally. Many are redundant.

Define novelty:

\[
N_t
=
\min_{s<t}
\|z_t-z_s\|
\]

Define quality:

\[
Q_t
=
\text{object confidence}
\times
\text{sharpness}
\times
\text{novelty}
\]

Keep the top \(K\) frames, for example:

\[
K=32,64,80
\]

For 64 views around 360°:

\[
360^\circ/64
=
5.625^\circ
\]

between selected views.

---

## 9.5 Pooling Into One Thumbprint

If each selected frame gives:

\[
z_t \in \mathbb{R}^{256}
\]

then compute:

\[
\mu = \frac{1}{K}\sum_t z_t
\]

\[
\sigma^2 = \frac{1}{K}\sum_t (z_t-\mu)^2
\]

\[
z_{\max}=\max_t z_t
\]

Then concatenate:

\[
T=[\mu,\sigma,z_{\max}]
\]

If \(z_t\) has 256 dimensions:

\[
T \in \mathbb{R}^{768}
\]

A richer thumbprint might be:

| Component | Dimensions |
|---|---:|
| Mean appearance | 256 |
| Appearance variance | 256 |
| Max/rare features | 256 |
| Shape statistics | 128 |
| Texture statistics | 128 |
| Scale-change statistics | 128 |
| View-consistency statistics | 128 |
| Mask-confidence statistics | 64 |
| Optional stereo-depth statistics | 256 |

Approximate result:

- monocular thumbprint: 1024–2048 dimensions
- stereo thumbprint: 1280–4096 dimensions

Recommended:

\[
\boxed{1024\text{–}2048\text{ real-valued dimensions}}
\]

---

## 9.6 Binary Hash Thumbprint

A real-valued thumbprint can be further compressed into a binary hash:

\[
h_i =
\begin{cases}
1 & a_i^\top T > b_i \\
0 & a_i^\top T \le b_i
\end{cases}
\]

This gives:

\[
H(V)\in\{0,1\}^{1024}
\]

A 1024-bit hash is:

\[
128 \text{ bytes}
\]

This is useful for:

- nearest-neighbor retrieval
- duplicate detection
- clustering similar objects
- fast pre-filtering

But for classification, the real-valued vector is usually better.

---

## 9.7 What Relevant Information Means

A patch is relevant if it has at least one of:

| Relevance type | Meaning |
|---|---|
| Object likelihood | probably belongs to centered object |
| Boundary value | helps determine shape |
| Class distinctiveness | helps distinguish object from others |
| View novelty | shows a new side |
| Scale novelty | shows detail unavailable elsewhere |
| Temporal consistency | persists across frames |
| Stereo consistency | has coherent depth |

The best score is therefore:

\[
\text{useful information}
=
\text{objectness}
\times
\text{distinctiveness}
\times
\text{nonredundancy}
\]

A flat blue center patch may be object but not very distinctive. A wrinkled boundary/finger patch may be much more informative.

---

# 10. Connection to Gaussian Splatting

Gaussian splatting is the natural 3D version of the compressed thumbprint idea.

The pipeline becomes:

\[
\text{video}
\rightarrow
\text{soft object evidence}
\rightarrow
\text{3D Gaussian object model}
\rightarrow
\text{compressed splat thumbprint}
\rightarrow
\text{classifier}
\]

Instead of storing pixels, represent the scene/object by 3D Gaussian primitives:

\[
G=\{g_1,g_2,\dots,g_N\}
\]

Each Gaussian has parameters:

\[
g_i=
(
\mu_i,\Sigma_i,\alpha_i,c_i,f_i,o_i
)
\]

where:

| Symbol | Meaning |
|---|---|
| \(\mu_i\in\mathbb{R}^3\) | 3D center |
| \(\Sigma_i\in\mathbb{R}^{3\times 3}\) | size, stretch, orientation |
| \(\alpha_i\) | opacity |
| \(c_i\) | color or appearance |
| \(f_i\) | learned feature vector |
| \(o_i\) | objectness probability |

The object mask becomes not just a 2D pixel mask, but a 3D objectness field over splats.

---

## 10.1 Center Prior as a Splat Prior

A Gaussian \(g_i\) projects into frame \(t\) as:

\[
\Pi_t(g_i)
\]

where \(\Pi_t\) is the camera projection.

The objectness of a Gaussian can be estimated by accumulating evidence over all views:

\[
o_i
=
\sigma
\left(
\sum_t
R_t(\Pi_t(\mu_i))
-
\lambda_{\text{bg}}B_i
\right)
\]

A simpler weighted average is:

\[
o_i
=
\frac{
\sum_t
v_{it}
P_{\text{center}}(\Pi_t(\mu_i))
w_e(r_{it})
A_{it}
}{
\sum_t v_{it}
}
\]

where:

- \(v_{it}\): Gaussian visibility in frame \(t\)
- \(r_{it}\): radius of projected Gaussian in frame \(t\)
- \(A_{it}\): appearance/object-likeness evidence

Thus:

\[
\boxed{\text{A splat is object-like if it repeatedly projects near the center and remains consistent across views.}}
\]

---

## 10.2 Mask Rendering From Object Splats

Once each Gaussian has objectness \(o_i\), the 2D mask for frame \(t\) is rendered:

\[
M_t(x,y)
=
\operatorname{Render}
\left(
\{(\mu_i,\Sigma_i,\alpha_i,o_i)\}_{i=1}^N
\right)
\]

This is better than estimating each mask independently:

\[
M_1,M_2,\dots,M_T
\]

because now all masks come from one shared 3D objectness model.

---

## 10.3 Weighted Reconstruction Loss

Radial information transport can be used inside the Gaussian-splat optimization:

\[
\mathcal{L}
=
\sum_{t,x,y}
w_e(r_{xy})
\left\|
I_t(x,y)-\hat I_t(x,y)
\right\|^2
\]

This forces the 3D model to care most about reconstructing the centered object.

But if \(e\) is too small, boundary splats are undertrained. Therefore:

\[
\boxed{e=0.32}
\]

is a good practical compromise.

---

## 10.4 Splat Thumbprint

Let the object splat set be:

\[
G_O=\{g_i:o_i>\tau\}
\]

Define per-splat features:

\[
\psi(g_i)
=
[
\mu_i,
\operatorname{eig}(\Sigma_i),
\alpha_i,
c_i,
f_i,
o_i
]
\]

Pool them:

\[
T
=
[
\operatorname{mean}_{i\in G_O}\psi(g_i),
\operatorname{var}_{i\in G_O}\psi(g_i),
\operatorname{max}_{i\in G_O}\psi(g_i),
\operatorname{hist}_{i\in G_O}\psi(g_i)
]
\]

Then compress:

\[
T_{\text{final}}=AT
\]

where \(A\) may be:

- PCA
- random projection
- learned linear layer
- autoencoder

Result:

\[
T_{\text{final}}\in\mathbb{R}^{1024}
\]

or:

\[
T_{\text{final}}\in\mathbb{R}^{2048}
\]

---

## 10.5 Compression Chain

For one 10-second HD video:

\[
1.87\text{ GB raw video}
\]

A possible Gaussian object representation:

- \(N=5{,}000\) object splats
- 48 fp16 numbers per splat

Size:

\[
5{,}000 \times 48 \times 2
=
480{,}000 \text{ bytes}
\]

So:

\[
1.87\text{ GB}
\rightarrow
0.48\text{ MB splat model}
\rightarrow
4\text{ KB classifier thumbprint}
\]

This is a natural hierarchy:

\[
\boxed{
\text{raw pixels}
\rightarrow
\text{3D object splats}
\rightarrow
\text{compact classifier vector}
}
\]

---

## 10.6 Why Gaussian Splatting Improves the Idea

The 2D method says:

\[
\text{this pixel is probably object in this frame}
\]

The Gaussian splatting method says:

\[
\text{this physical 3D blob is probably part of the object}
\]

This is much stronger.

If a glove finger is ambiguous in one frame but clear in another, the 3D Gaussian representing that finger can accumulate object evidence from the clear views and still render correctly into the ambiguous views.

Thus:

\[
\boxed{
\text{Gaussian splatting converts multi-frame statistical mask inference into multi-view physical object inference.}
}
\]

---

## 10.7 Stereo and Gaussian Splatting

Stereo helps Gaussian splatting because it provides depth initialization:

\[
(x,y,d)
\rightarrow
(X,Y,Z)
\]

Monocular video can still work using structure-from-motion, but stereo is more stable.

For this system:

\[
\boxed{
\text{stereo 360° video}
\rightarrow
\text{better depth}
\rightarrow
\text{better splats}
\rightarrow
\text{better thumbprint}
}
\]

---

# 11. Best Practical System

The best practical pipeline is:

\[
\boxed{
\text{centered 360° stereo video}
\rightarrow
\text{soft mask inference}
\rightarrow
\text{weighted Gaussian splat reconstruction}
\rightarrow
\text{object-splat pruning}
\rightarrow
\text{compressed thumbprint}
\rightarrow
\text{1024-neuron FFNN classifier}
}
\]

If Gaussian splatting is too heavy, use:

\[
\boxed{
\text{centered monocular/stereo video}
\rightarrow
\text{soft relevance masks}
\rightarrow
\text{nonredundant view selection}
\rightarrow
\text{pooled object features}
\rightarrow
\text{1024-neuron FFNN}
}
\]

Recommended edge retention:

\[
\boxed{32\%}
\]

or, if bandwidth is constrained:

\[
\boxed{16\%}
\]

Avoid:

\[
\boxed{2\%,4\%,8\%}
\]

unless the object remains small and well inside the high-information center.

---

# 12. Failure Modes and Limits

The method can fail or degrade when:

- the object is not actually centered
- the object leaves the frame
- the object fills the frame too much while edge retention is low
- the background is always close to the center
- the object is transparent or reflective
- object color matches background
- the object is thin, like wire or string
- motion blur is high
- lighting changes quickly
- shadows move with the object
- stereo disparity fails on textureless regions
- monocular pose estimation fails
- the camera path is not known or not recoverable
- the system confuses stable background with object

Important uncertainty measure:

\[
H_t(u)
=
-q_t(u)\log q_t(u)
-
(1-q_t(u))\log(1-q_t(u))
\]

High entropy means the mask is uncertain.

The system should preserve soft masks and confidence values rather than only hard masks.

---

# 13. Implementation Sketch

## 13.1 Non-Gaussian Thumbprint Pipeline

```python
def extract_thumbprint(video, edge_retention=0.32, stereo=None):
    frames = sample_frames(video, max_frames=300)

    # 1. Compute radial center prior and information reliability
    center_prior = compute_center_prior(frames)
    radial_weight = compute_radial_weight(frames, edge_retention)

    # 2. Initialize soft masks
    q = center_prior * radial_weight

    # 3. Iterative EM-style refinement
    for _ in range(5):
        object_model = fit_feature_model(frames, weights=q)
        background_model = fit_feature_model(frames, weights=1-q)

        appearance_score = log_likelihood_ratio(
            frames,
            object_model,
            background_model
        )

        consistency_score = compute_multiframe_consistency(frames, q)

        if stereo is not None:
            depth_score = compute_depth_objectness(stereo, q)
        else:
            depth_score = 0

        logits = (
            logit(center_prior)
            + radial_weight * appearance_score
            + radial_weight * consistency_score
            + depth_score
        )

        q = sigmoid(logits)
        q = spatial_temporal_smooth(q)

    # 4. Extract relevance-weighted per-frame features
    z = []
    for frame, mask in zip(frames, q):
        features = relevance_weighted_features(frame, mask)
        z.append(features)

    # 5. Select nonredundant informative views
    selected = select_by_quality_novelty(z, q, k=64)

    # 6. Pool into a single thumbprint
    T = pool_features(selected, stats=["mean", "var", "max"])

    # 7. Compress to classifier size
    T = project_to_dimension(T, dim=2048)

    return T, q
```

---

## 13.2 Gaussian-Splat Thumbprint Pipeline

```python
def extract_gaussian_splat_thumbprint(video, edge_retention=0.32, stereo=None):
    frames = sample_frames(video)

    # 1. Estimate camera poses and depth
    if stereo is not None:
        depth = estimate_stereo_depth(stereo)
        poses = estimate_or_use_known_camera_path(frames, depth)
    else:
        poses, sparse_points = structure_from_motion(frames)
        depth = None

    # 2. Compute radial weights
    radial_weight = compute_radial_weight(frames, edge_retention)

    # 3. Initialize Gaussian splats
    gaussians = initialize_gaussians(frames, poses, depth)

    # 4. Optimize splats using center-weighted reconstruction loss
    gaussians = optimize_gaussians(
        gaussians,
        frames,
        poses,
        pixel_weights=radial_weight
    )

    # 5. Estimate per-Gaussian objectness
    objectness = estimate_splat_objectness(
        gaussians,
        frames,
        poses,
        center_prior=True,
        radial_weight=radial_weight
    )

    # 6. Keep object-like splats
    object_splats = [
        g for g, o in zip(gaussians, objectness)
        if o > 0.5
    ]

    # 7. Pool splat statistics into a vector
    T = pool_splat_features(
        object_splats,
        stats=["mean", "var", "max", "hist"]
    )

    # 8. Compress
    T = project_to_dimension(T, dim=2048)

    return T, object_splats, objectness
```

---

# 14. Summary

The central concept is:

\[
\boxed{
\text{The centered video is a weakly supervised object-information channel.}
}
\]

Because the object is always kept near the center, the center prior can act as a weak label. Across many frames, this weak prior can be refined into a soft object mask.

The full chain is:

\[
\text{HD video}
\rightarrow
\text{radial information weighting}
\rightarrow
\text{soft mask inference}
\rightarrow
\text{object-relevance extraction}
\rightarrow
\text{compressed thumbprint}
\rightarrow
\text{classifier}
\]

Stereo improves the chain by adding depth:

\[
\text{RGB}
+
\text{depth}
\rightarrow
\text{better masks}
\rightarrow
\text{better 3D shape}
\rightarrow
\text{better classification}
\]

Gaussian splatting improves it further by converting frame-wise mask inference into 3D object inference:

\[
\text{many 2D soft masks}
\rightarrow
\text{one 3D objectness field}
\]

The best version is:

\[
\boxed{
\text{stereo 360° video}
+
32\%\text{ edge retention}
+
\text{soft mask inference}
+
\text{Gaussian splat object model}
+
\text{2048-dimensional thumbprint}
+
\text{1024-neuron FFNN}
}
\]

For practical non-3D use:

\[
\boxed{
\text{monocular/stereo video}
+
32\%\text{ edge retention}
+
\text{soft relevance masks}
+
\text{pooled multi-view features}
+
\text{1024–2048-dimensional thumbprint}
}
\]

---

# 15. References and Related Concepts

These are conceptual anchors for the ideas discussed.

## Information Bottleneck

The thumbprint objective resembles the information bottleneck principle:

\[
\max_T I(T;Y)-\lambda I(T;X)
\]

That is, compress \(X\) while preserving information about \(Y\).

## Multi-View Recognition

The 360° video acts like multi-view recognition: multiple views reduce ambiguity and expose different object surfaces.

## Stereo Depth

Stereo contributes most at close distances because disparity precision degrades roughly quadratically with distance.

## Gaussian Splatting

3D Gaussian Splatting represents a scene as many explicit 3D Gaussians that can be rendered into camera views. In this object-centered setting, Gaussian splats can become a compressed 3D object representation.

Useful terms to search:

- 3D Gaussian Splatting
- object-centric 3D reconstruction
- Gaussian splatting segmentation
- multi-view object recognition
- RGB-D object classification
- information bottleneck
- weakly supervised segmentation
- co-segmentation from video
- self-supervised object discovery
- next-best-view recognition
- neural object descriptors
- binary hashing for image retrieval

---

# Appendix A: Key Formulas

## Radial Information Retention

\[
w_e(r)
=
e+(1-e)
\frac{e^{-2r^2}-e^{-2}}{1-e^{-2}}
\]

## Center Prior

\[
\pi(r)=e^{-\frac{r^2}{2\sigma_c^2}}
\]

## Mask Posterior

\[
q_t(u)
=
\sigma
\left(
\log \frac{\pi(r_u)}{1-\pi(r_u)}
+
\lambda_A w_e(r_u) A_t(u)
+
\lambda_C w_e(r_u) C_t(u)
+
\lambda_S S_t(u)
\right)
\]

## Relevance Score

\[
R_t(u)
=
P_{\text{center}}(u)
\cdot
w_e(r(u))
\cdot
P(\text{object-like}\mid f_t(u))
\cdot
P(\text{temporally consistent})
\cdot
\text{novelty}
\]

## Frame Feature

\[
z_t
=
\frac{
\sum_u R_t(u)\phi(I_t,u)
}{
\sum_u R_t(u)
}
\]

## Thumbprint

\[
T(V)
=
\operatorname{compress}
\left(
\sum_{t,u}
R_t(u)\phi(I_t,u)
\right)
\]

## FFNN

\[
\hat y
=
\operatorname{softmax}
\left(
W_2\phi(W_1x+b_1)+b_2
\right)
\]

## Splat Objectness

\[
o_i
=
\frac{
\sum_t
v_{it}
P_{\text{center}}(\Pi_t(\mu_i))
w_e(r_{it})
A_{it}
}{
\sum_t v_{it}
}
\]

## Splat Thumbprint

\[
T_{\text{final}}
=
A
[
\operatorname{mean}_{i\in G_O}\psi(g_i),
\operatorname{var}_{i\in G_O}\psi(g_i),
\operatorname{max}_{i\in G_O}\psi(g_i),
\operatorname{hist}_{i\in G_O}\psi(g_i)
]
\]

---

# Appendix B: Recommended Defaults

| Component | Recommendation |
|---|---|
| Edge retention | 32% |
| Conservative edge retention | 64% |
| Minimum practical edge retention | 16% |
| Avoid unless object stays small | 8%, 4%, 2% |
| Number of selected views | 32–80 |
| Thumbprint size | 1024–2048 real values |
| Binary retrieval hash | 1024 bits |
| Classifier input | thumbprint, not raw video |
| FFNN | 1024 hidden neurons, 512 outputs |
| Best capture | stereo 360° video from 10 m to 0.5 m |
| Best 3D representation | Gaussian splat object model |

---

# Appendix C: Practical Interpretation for a Blue Rubber Glove

For a blue rubber glove, the most useful object evidence is:

- blue color dominance
- rubber-like smooth reflectance
- soft folds and wrinkles
- finger-like protrusions
- thin boundary
- crumpled 3D geometry
- low physical thickness
- boundary curvature
- depth separation from grass/road if stereo is available

A good thumbprint should discard:

- grass texture
- road texture
- far background
- shadows not attached to the object
- repeated near-identical frames
- low-confidence edge evidence
- blur

For this object type, very aggressive edge compression is risky because the glove identity may depend strongly on boundary/finger shape.

Recommended setting:

\[
\boxed{32\%\text{ edge retention, stereo if possible, Gaussian splat thumbprint if feasible.}}
\]
