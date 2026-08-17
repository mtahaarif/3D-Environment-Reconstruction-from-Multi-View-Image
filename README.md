# 3D Environment Reconstruction from 2D Images

**Classical SfM / MVS + 3D Gaussian Splatting on the DTU Robot Image Dataset**
**Platform** | Kaggle GPU Notebook (NVIDIA T4) |
| **Core stack** | Python · PyTorch · OpenCV · Open3D · gsplat · plyfile · SciPy |

----

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Facts](#2-quick-facts)
3. [Background & Theory](#3-background--theory)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Repository / Notebook Structure](#5-repository--notebook-structure)
6. [Environment & Dependencies](#6-environment--dependencies)
7. [Dataset](#7-dataset)
8. [Configuration Reference](#8-configuration-reference)
9. [Stage-by-Stage Walkthrough](#9-stage-by-stage-walkthrough)
   - [9.1 Camera Calibration Loading](#91-camera-calibration-loading)
   - [9.2 Image Loading & Preprocessing](#92-image-loading--preprocessing)
   - [9.3 SIFT Feature Extraction](#93-sift-feature-extraction)
   - [9.4 Feature Matching (FLANN + Lowe + RANSAC)](#94-feature-matching-flann--lowe--ransac)
   - [9.5 Two-View Triangulation — Sparse SfM Cloud](#95-two-view-triangulation--sparse-sfm-cloud)
   - [9.6 MVS Densification — Plane-Sweep Stereo](#96-mvs-densification--plane-sweep-stereo)
   - [9.7 Dense Cloud Cleanup](#97-dense-cloud-cleanup)
   - [9.8 3D Gaussian Splatting](#98-3d-gaussian-splatting)
   - [9.9 Export & Interactive Viewers](#99-export--interactive-viewers)
10. [Mathematical Models Reference](#10-mathematical-models-reference)
11. [How to Run](#11-how-to-run)
12. [Output Files](#12-output-files)
13. [Results Summary](#13-results-summary)
14. [Known Issues, Limitations & Design Notes](#14-known-issues-limitations--design-notes)
15. [Future Work](#15-future-work)
16. [Conclusion](#16-conclusion)
17. [Credits & References](#17-credits--references)

---

## 1. Project Overview

This project implements a **complete, end-to-end 3D reconstruction pipeline** that turns a set of ordinary 2D photographs into a fully navigable 3D representation of a real-world object. It was built as the Computer Vision (CEP) course project and combines two generations of computer vision technique in a single pipeline:

- **Classical geometry-based reconstruction** — Structure-from-Motion (SfM) and Multi-View Stereo (MVS), using hand-crafted features (SIFT), robust matching (FLANN + RANSAC), and closed-form geometric solvers (DLT triangulation, plane-sweep stereo).
- **Modern differentiable rendering** — 3D Gaussian Splatting (3DGS, 2023), which represents the scene as thousands of optimizable 3D Gaussians and learns them end-to-end via gradient descent against the input photos.

The two are **not run independently** — the classical stage's cleaned point cloud is used to *initialize* the Gaussian Splatting optimizer, so the modern differentiable method starts from a geometrically correct scaffold rather than random noise or SfM-only points. This is the central design idea of the project.

The pipeline is applied to **scan1** of the **DTU Robot Image Dataset**, a benchmark dataset used widely in multi-view stereo research, reconstructing a decorative tin can object photographed from 49 calibrated camera positions arranged in a hemispherical arc.

The entire project lives in a single, heavily documented Jupyter notebook (`3D_Reconstruction.ipynb`) designed to run top-to-bottom on a Kaggle GPU instance, and produces four progressively higher-fidelity 3D outputs plus self-contained, installation-free interactive 3D viewers for each.

---

## 2. Quick Facts

| Item | Value |
|---|---|
| Dataset | DTU Robot Image Dataset — SampleSet, `scan1`, lighting = 3, resolution `r5000` |
| Input views | 49 rectified images (of 64 available camera positions), downscaled 50% → ~800×600 px |
| Sparse (SfM) cloud | 30,583 points — SIFT + FLANN matching + two-view DLT triangulation |
| Dense (MVS) cloud | 166,303 points — plane-sweep stereo with NCC scoring |
| Cleaned dense cloud | 130,029 points — DTU ObsMask + statistical + radius outlier removal (21.8% noise removed) |
| Gaussian Splat | ~70,000–130,000 Gaussians, trained for 15,000 iterations |
| Final quality | ~25 dB PSNR at convergence, ~0.04 combined L1+SSIM loss |
| Compute | Kaggle GPU notebook, NVIDIA T4 |
| Key libraries | PyTorch, `gsplat`, OpenCV (`opencv-contrib-python`), Open3D, `plyfile`, SciPy |

---

## 3. Background & Theory

### 3.1 The problem

3D reconstruction from 2D images is the task of recovering the 3D shape, geometry, and appearance of a real scene from a finite collection of photographs. It underlies applications in autonomous robotics and SLAM, augmented/virtual reality, cultural-heritage digitization, surgical planning, and scene understanding for autonomous vehicles.

### 3.2 Structure from Motion (SfM)

SfM simultaneously recovers **3D scene geometry** and **camera poses** from a set of overlapping images, without any prior knowledge of scene structure. The classical pipeline is:

1. Detect distinctive, repeatable local features in every image.
2. Match features across image pairs.
3. Apply the epipolar geometric constraint to reject false matches.
4. Triangulate the 3D position of every surviving matched point.

The output is a **sparse 3D point cloud** plus (in general SfM) recovered camera poses — though in this project the camera poses are already known from the DTU calibration files, so SfM here is used purely for triangulating 3D structure, not for pose estimation.

### 3.3 Multi-View Stereo (MVS)

MVS takes the sparse SfM result and **densifies** it by estimating a per-pixel depth map for every image. This project uses **plane-sweep stereo**: for every pixel of a reference image, a bank of depth hypotheses is tested; for each hypothesis, the corresponding pixel patch is reprojected into several neighboring views and scored for photo-consistency using **Normalized Cross-Correlation (NCC)**, which is invariant to per-view illumination differences. The depth with the best NCC score (above a threshold, and confirmed by enough neighbouring views) is kept. All accepted depths across all reference views are fused into one dense point cloud.

### 3.4 3D Gaussian Splatting (3DGS)

3DGS (Kerbl et al., 2023) is a real-time, differentiable, explicit scene representation for novel-view synthesis. The scene is modeled as a large collection of anisotropic 3D Gaussians, each with five learnable parameter groups:

- **Position** (μ, 3D)
- **Anisotropic scale** (3D, stored in log-space)
- **Rotation** (quaternion)
- **Opacity** (stored in inverse-sigmoid/logit space)
- **Color** (RGB, or spherical-harmonic coefficients)

A **differentiable, tile-based rasterizer** projects and alpha-composites all Gaussians into a rendered 2D image for a given camera pose. This render is compared against the ground-truth photo with a photometric loss, and gradients are backpropagated directly into the Gaussian parameters — no neural network is required. An adaptive **densification loop** (clone / split / prune) periodically grows or shrinks the Gaussian population during training to better fit under- and over-represented regions of the scene.

---

## 4. Pipeline Architecture

The system is organized as **six sequential stages**. Data flows from raw calibrated 2D images, through classical geometric reconstruction (SfM → MVS → cleanup), into a differentiable optimization stage (3DGS), and finally to exported, viewer-ready assets.

| Stage | Input | Key Algorithm / Tool | Output |
|---|---|---|---|
| 1 — Feature Extraction | 49 grayscale images | SIFT, 8,000 features/image (OpenCV) | Keypoints + 128-D descriptors per view |
| 2 — Feature Matching | Descriptor pairs (views within ±12 index distance) | FLANN k‑NN + Lowe's ratio test + RANSAC (fundamental matrix) | Inlier match database (pair → matches) |
| 3 — SfM Triangulation | Inlier matches + camera projection matrices P | `cv2.triangulatePoints` (DLT) + cheirality/reprojection filtering | Sparse cloud: 30,583 points |
| 4 — MVS Densification | Images + calibrated cameras | Plane-sweep stereo, NCC scoring, 96 depth hypotheses, stride 4 | Dense cloud: 166,303 points |
| 5 — Cloud Cleanup | Dense cloud + DTU ObsMask | Dilated observability mask filter + Statistical Outlier Removal (SOR, k=20) + Radius filter (r=15 mm) | Cleaned cloud: 130,029 points |
| 6 — Gaussian Splatting | Cleaned cloud + 49 ground-truth images (GPU) | `gsplat` differentiable rasterizer + Adam optimizer + clone/split/prune densification | Optimized 3DGS scene (`.ply`, ~70k–130k Gaussians) |

Every intermediate representation (sparse, dense, cleaned, and the final Gaussian Splat) is exported as both a `.ply` point-cloud file **and** a self-contained Three.js HTML viewer for interactive rotate/zoom/pan inspection with zero software installation.

```
 49 Calibrated Images ──► SIFT ──► FLANN+RANSAC Matching ──► DLT Triangulation
                                                                    │
                                                                    ▼
                                                          Sparse Cloud (30,583 pts)
                                                                    │
                                                     Plane-Sweep MVS (NCC scoring)
                                                                    │
                                                                    ▼
                                                          Dense Cloud (166,303 pts)
                                                                    │
                                            ObsMask + SOR + Radius Outlier Removal
                                                                    │
                                                                    ▼
                                                        Cleaned Cloud (130,029 pts)
                                                                    │
                                        Gaussian Init → Differentiable Rasterize
                                        → L1+SSIM Loss → Clone/Split/Prune → Adam
                                                                    │
                                                                    ▼
                                          3D Gaussian Splat (~70k–130k Gaussians)
                                                                    │
                                                                    ▼
                                         .ply exports + Three.js interactive viewers
```

---

## 5. Repository / Notebook Structure

The project is a **single Jupyter notebook**: `3D_Reconstruction.ipynb`, organized into 14 numbered sections (markdown headers followed by code cells), designed to be run sequentially top to bottom:

| # | Section | Purpose |
|---|---|---|
| 1 | Install Dependencies | pip-installs all required libraries, including a CUDA compile of `gsplat` |
| 2 | Imports & GPU Check | imports, fixed random seeds, device/GPU detection |
| 3 | Configuration | all dataset paths and pipeline hyperparameters in one place, plus the shared HTML-viewer generator |
| 4 | Load Projection Matrices | parses DTU calibration files, decomposes into K, R, t |
| 5 | Visualize Camera Positions in 3D | sanity-check plot of the 49 camera poses |
| 6 | Load Images | grayscale + RGB loading, downscaling |
| 7 | SIFT Feature Extraction | keypoint + descriptor computation for all views |
| 8 | Feature Matching | FLANN + Lowe + RANSAC pairwise matching, plus best-pair and match-graph visualizations |
| 9 | Triangulation (Sparse Cloud) | DLT triangulation with cheirality/reprojection filtering, dedup + outlier removal, sparse cloud plots |
| 10 | MVS Densification | depth-range estimation, neighbour selection, plane-sweep stereo core |
| 10b | Dense Cloud Cleanup | DTU ObsMask + SOR + radius filter, before/after comparison plots |
| 11 | 3D Gaussian Splatting | Gaussian init, differentiable rendering, densification loop, training loop, training-curve and render-progression plots |
| 12 | Export Gaussian Splatting `.ply` | SuperSplat-compatible `.ply` writer |
| 13 | Interactive HTML Viewers | generates Three.js viewers for all four representations |
| 14 | Summary | recap of outputs, pipeline stages, and demo instructions |

Because Kaggle notebooks are self-contained, there is **no separate `src/` package** — all functions (camera decomposition, image loading, SIFT wrapper, matcher, triangulator, plane-sweep MVS, cleanup filters, Gaussian init/render/densify/optimize, `.ply`/HTML export) are defined inline in the notebook's code cells in the order they are used.

A companion Word document, `3D_Reconstruction_Report.docx`, is the formal written project report submitted for the course; it walks through the same pipeline with embedded code excerpts, 13 result figures, and the mathematical derivations summarized in this README.

---

## 6. Environment & Dependencies

The notebook is written to run on **Kaggle's GPU runtime** (NVIDIA T4), but any environment with an NVIDIA GPU, CUDA, and Python 3.10+ will work.

**Section 1 of the notebook installs:**

```python
pip('opencv-contrib-python==4.10.0.84',
    'plyfile==1.0.3', 'open3d', 'tqdm', 'matplotlib', 'scipy', 'pillow')
pip('gsplat')   # ~3–5 minute CUDA compile on first run
```

**Full dependency list:**

| Package | Role |
|---|---|
| `torch` (+ CUDA) | Tensor ops, autograd, Adam optimizer for Gaussian Splatting training |
| `gsplat` | Differentiable, tile-based CUDA rasterizer for 3D Gaussian Splatting (`gsplat.rendering.rasterization`) |
| `opencv-contrib-python==4.10.0.84` | SIFT feature detection, FLANN matching, fundamental-matrix RANSAC, DLT triangulation, image I/O |
| `open3d` | Point-cloud data structures, voxel downsampling, statistical/radius outlier removal |
| `plyfile==1.0.3` | Reading/writing `.ply` point cloud and Gaussian Splat files |
| `scipy` | `scipy.io` (loading DTU's `.mat` ObsMask), `scipy.ndimage.binary_dilation`, `scipy.spatial.cKDTree` |
| `numpy` | Core numerical arrays throughout |
| `matplotlib` | All static visualizations (camera plots, clouds, match graphs, training curves) |
| `tqdm` | Progress bars for the 15,000-iteration training loop |
| `pillow` | Image utility support |

Implicit standard-library usage: `os`, `glob`, `time`, `json`, `math`, `warnings`, `random`, `pathlib.Path`.

Reproducibility is enforced with fixed seeds: `np.random.seed(42)`, `torch.manual_seed(42)`, `random.seed(42)`.

---

## 7. Dataset

**DTU Robot Image Dataset** (SampleSet), a well-known multi-view stereo benchmark captured with a robot-mounted structured-light scanner and camera rig.

- **Scan used:** `scan1` — a decorative tin can (with lettering "OLIVE," a red wax-seal-style top, and a golden rim) standing on a brick-textured tile, photographed in a controlled black-background studio setup.
- **Camera positions:** 64 total pre-calibrated positions arranged in a roughly hemispherical arc around the object; **49 are used** in this project (`VIEW_IDS = range(1, 50)`).
- **Lighting condition:** `3` (of the multiple lighting conditions DTU provides per scan).
- **Resolution tag:** `r5000` (DTU's rectified image naming convention); images are then downscaled by `IMG_SCALE = 0.5` to roughly 800×600 px for a speed/quality tradeoff.
- **Calibration:** DTU ships per-camera 3×4 projection matrices in plain-text files (`pos_XXX.txt`, calibration set `cal18`), from which intrinsics (K), rotation (R), and translation (t) are recovered via `cv2.decomposeProjectionMatrix`.
- **Observability mask (ObsMask):** DTU also supplies a 3D binary voxel volume (loaded from a `.mat` file via `scipy.io`) marking the valid/observed region of space around the object — used in the cleanup stage to strip background and turntable points.

Expected on-disk path structure (as configured for the Kaggle environment):

```
/kaggle/input/datasets/<user>/3d-reconstruction-dataset/data/SampleSet/MVS Data/
├── Calibration/cal18/pos_XXX.txt        # 64 projection matrices
└── Rectified/scan1/rect_XXX_3_r5000.png # 49 rectified images used
```

To run outside Kaggle, update `DATA_ROOT` in the Configuration section (Section 3 of the notebook) to point at a local copy of the DTU SampleSet with the same folder layout.

---

## 8. Configuration Reference

All tunable parameters live in one place (Section 3 of the notebook). This is the authoritative "control panel" for the whole pipeline:

**Dataset / views**
```python
DATA_ROOT  = Path('.../3d-reconstruction-dataset/data/SampleSet/MVS Data')
CALIB_DIR  = DATA_ROOT / 'Calibration' / 'cal18'
SCAN       = 'scan1'
RECT_DIR   = DATA_ROOT / 'Rectified' / SCAN
LIGHTING   = '3'
RESOLUTION = 'r5000'
N_VIEWS    = 49
IMG_SCALE  = 0.5          # → ~800×600 px
```

**SfM hyperparameters**
```python
SIFT_FEATURES  = 8000     # max SIFT detections per image
MATCH_RATIO    = 0.8      # Lowe's ratio test threshold
MIN_MATCHES    = 25       # minimum matches to attempt RANSAC on a pair
RANSAC_REPROJ  = 4.0      # px, fundamental-matrix RANSAC threshold
MAX_PAIR_DIST  = 12       # only match views within ±12 index positions
MIN_TRACK_LEN  = 2        # accept two-view tracks (not requiring 3+ view agreement)
MAX_REPROJ_ERR = 6.0      # px, triangulated-point reprojection filter
```

**MVS hyperparameters**
```python
MVS_PATCH      = 7        # 7×7 NCC patch
MVS_DEPTHS     = 96       # depth hypotheses per pixel
MVS_STRIDE     = 4        # sample every 4th pixel
MVS_MIN_VIEWS  = 2        # require agreement from ≥2 neighbouring views
MVS_NCC_THRESH = 0.3      # accept depth if NCC score > 0.3
MVS_REF_STRIDE = 3        # use every 3rd view as an MVS reference view
```

**Gaussian Splatting hyperparameters**
```python
GS_ITERS         = 15000
GS_LR_POS_INIT   = 0.00016
GS_LR_POS_FINAL  = 0.0000016   # exponential decay over training
GS_LR_OPACITY    = 0.05
GS_LR_SCALE      = 0.005
GS_LR_ROT        = 0.001
GS_LR_COLOR      = 0.0025

# Densification loop
GS_DENSIFY_FROM  = 500
GS_DENSIFY_UNTIL = 7500
GS_DENSIFY_EVERY = 100
GS_OPACITY_RESET = 3000
GS_GRAD_THRESH   = 0.0002
GS_OPACITY_PRUNE = 0.005
GS_SCALE_PRUNE   = 0.1          # fraction of scene extent

GS_LOG_EVERY     = 250
GS_RENDER_EVERY  = 1500
```

**Output**
```python
OUT_DIR = Path('/kaggle/working/output')
# subfolders: figures/, renders/
```

Several parameters carry inline comments in the notebook noting they were tuned from an earlier "v1" version (e.g., `MATCH_RATIO` was 0.75 → 0.8, `MAX_PAIR_DIST` was 8 → 12, `MVS_PATCH` was 11 → 7, `GS_ITERS` was 7000 → 15000) — see [Section 14 — Known Issues, Limitations & Design Notes](#14-known-issues-limitations--design-notes) for what these fixes addressed.

---

## 9. Stage-by-Stage Walkthrough

### 9.1 Camera Calibration Loading

*(Notebook Section 4)*

DTU provides pre-calibrated 3×4 projection matrices **P**, one per camera, as plain-text files. Each matrix encodes the complete mapping from 3D world coordinates to 2D image pixels:

```
P = K [R | t]        x ∼ P·X
```

`cv2.decomposeProjectionMatrix` factors P into the intrinsic matrix **K** (focal lengths, principal point, skew), rotation **R**, and translation **t**:

```python
def decompose_projection(P):
    K, R, t_homog = cv2.decomposeProjectionMatrix(P)[:3]
    t = t_homog[:3] / t_homog[3]      # de-homogenise
    K = K / K[2, 2]                   # enforce K[2,2] = 1
    C = t.copy()                      # camera centre in world coords
    t_wc = -R @ C                     # world-to-camera translation
    return K, R, t_wc, C
```

Since images are downscaled 50%, the intrinsics are scaled to match:
```python
K_s[0,0] *= 0.5;  K_s[1,1] *= 0.5
K_s[0,2] *= 0.5;  K_s[1,2] *= 0.5
```

A 3D + top-down plot (Section 5) confirms all 49 cameras form a hemispherical arc pointed inward at a shared scene centre — a correctness check before any reconstruction begins.

### 9.2 Image Loading & Preprocessing

*(Notebook Section 6)*

Images are loaded with OpenCV, converted to both RGB (visualization / Gaussian training targets) and grayscale (feature extraction), and resized to 50% using `INTER_AREA` interpolation — chosen because it area-averages pixel blocks during downscaling rather than point-sampling, which reduces aliasing:

```python
def load_image(view_id, lighting=LIGHTING, scale=0.5):
    fname = f'rect_{view_id:03d}_{lighting}_{RESOLUTION}.png'
    img = cv2.imread(str(RECT_DIR / fname))
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w*scale), int(h*scale)),
                      interpolation=cv2.INTER_AREA)
    return img
```

### 9.3 SIFT Feature Extraction

*(Notebook Section 7)*

SIFT (Scale-Invariant Feature Transform) detects local features invariant to scale, rotation, and moderate illumination change, via: (1) Difference-of-Gaussian (DoG) scale-space extrema detection, (2) sub-pixel keypoint localization, (3) dominant-gradient orientation assignment, (4) a 128-D gradient-histogram descriptor.

```
DoG(x, y, σ) = [G(x,y,kσ) − G(x,y,σ)] * I(x,y)
```

```python
sift = cv2.SIFT_create(
    nfeatures=8000,
    contrastThreshold=0.04,   # reject low-contrast keypoints
    edgeThreshold=10,         # suppress edge responses (Harris ratio)
)
kp, des = sift.detectAndCompute(images_gray[vid], None)
```

Keypoints concentrate on textured regions (surface lettering, decorative elements, the rough brick tile); smooth or shadowed regions and the black studio background yield very few (some far-side views detect under 1,000 keypoints).

### 9.4 Feature Matching (FLANN + Lowe + RANSAC)

*(Notebook Section 8)*

A three-stage filter cascade removes incorrect correspondences — each stage targets a distinct failure mode: descriptor ambiguity, random geometric coincidence, and excessive view dissimilarity.

1. **FLANN k-NN matching (k=2):** KD-tree-indexed approximate nearest-neighbour search in 128-D descriptor space — far faster than brute force.
2. **Lowe's ratio test:** keep match (m, n) only if `m.distance < 0.8 × n.distance`, rejecting ambiguous matches.
3. **Fundamental-matrix RANSAC:** estimate F between the surviving matches, keep only geometric inliers satisfying the epipolar constraint within 4 px. Only view pairs within `MAX_PAIR_DIST = 12` index positions are attempted at all (adjacent turntable positions overlap the most).

```
x'ᵀ F x = 0        (epipolar constraint)
```

```python
flann = cv2.FlannBasedMatcher({'algorithm':1,'trees':5}, {'checks':50})

def match_pair(vid_a, vid_b):
    raw = flann.knnMatch(descriptors[vid_a], descriptors[vid_b], k=2)
    good = [m for (m, n) in raw if m.distance < 0.8 * n.distance]
    if len(good) < 25:
        return []
    pts_a = np.float32([kp[vid_a][m.queryIdx].pt for m in good])
    pts_b = np.float32([kp[vid_b][m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(pts_a, pts_b,
                cv2.FM_RANSAC, ransacReprojThreshold=4.0)
    return [g for g, ok in zip(good, mask.ravel()) if ok]
```

The best matched pair in the dataset (views 009↔010) produced 1,134 verified inlier correspondences. A pairwise match-count heatmap over all views shows a bright diagonal band of width ≈12 (the `MAX_PAIR_DIST` constraint), with the strongest matches (1,000+) between near-adjacent turntable positions, and essentially zero spurious off-diagonal matches — confirming the RANSAC filter is working correctly.

### 9.5 Two-View Triangulation — Sparse SfM Cloud

*(Notebook Section 9)*

For every matched image pair, 3D points are recovered via the **Direct Linear Transform (DLT)**: given 2D correspondences and known projection matrices, a linear system `AX = 0` is built from cross-products of the projection equations, and solved via SVD (the 3D point is the last right singular vector of A).

```
A = [xₐ × Pₐ ;  x_b × P_b]        AX = 0  →  X = last right singular vector of A
err = ‖x − π(PX)‖₂
```

```python
def triangulate_two_view(vid_a, vid_b, matches):
    P_a, P_b = cameras[vid_a]['P_s'], cameras[vid_b]['P_s']
    pts_a = np.float32([kp[vid_a][m.queryIdx].pt for m in matches]).T
    pts_b = np.float32([kp[vid_b][m.trainIdx].pt for m in matches]).T
    pts_4d = cv2.triangulatePoints(P_a, P_b, pts_a, pts_b)
    return (pts_4d[:3] / pts_4d[3]).T   # de-homogenise: (N, 3)
```

Per-point quality filters reject:
- **Cheirality failures** — points not in front of both cameras.
- **High reprojection error** — worse than 6.0 px in either view.

Because the same physical point is triangulated independently from multiple overlapping pairs, the raw output is deduplicated with a 1 mm voxel grid merge and cleaned with a statistical outlier filter:

```python
pcd = pcd.voxel_down_sample(voxel_size=1.0)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
```

**Result: 30,583 sparse points.** The object is clearly identifiable in the resulting cloud, but lacks fine surface detail — this is expected of sparse SfM and is exactly what the MVS stage addresses next.

### 9.6 MVS Densification — Plane-Sweep Stereo

*(Notebook Section 10)*

Plane-sweep stereo tests many depth hypotheses per pixel and keeps the one with the strongest photo-consistency across neighbouring views, scored with **Normalized Cross-Correlation (NCC)** — chosen for its illumination-invariance, important given DTU's per-view lighting variation.

```
NCC(p, d) = Σ_q [(r(q) − μ_r)/σ_r] · [(n(π(d,q)) − μ_n)/σ_n] / N
```

Key steps in `plane_sweep_mvs(...)`:

1. **Depth range estimation** from the sparse cloud (5th–95th percentile of per-camera point depths).
2. **Inverse-depth sampling** — depth hypotheses are spaced uniformly in 1/d, giving finer resolution near the camera:
   ```python
   inv_n, inv_f = 1.0/near, 1.0/far
   depth_samples = 1.0 / np.linspace(inv_n, inv_f, 96)
   ```
3. For each pixel and each depth hypothesis: back-project to a 3D ray → transform to world space → reproject into each neighbouring view → extract a 7×7 patch → compute NCC against the reference patch.
4. Accept the depth with the best NCC score if it exceeds `MVS_NCC_THRESH = 0.3` and is confirmed by at least `MVS_MIN_VIEWS = 2` neighbours.
5. Every 3rd view (`MVS_REF_STRIDE = 3`) is used as a reference view, with its `k=4` nearest neighbouring cameras (by camera-centre distance) used for scoring.

**Result: 166,303 dense points** — roughly 5.4× the sparse count, with the object surface now filled in with recognizable detail. The raw dense cloud still contains a visible noise halo from the turntable/background, addressed next.

### 9.7 Dense Cloud Cleanup

*(Notebook Section 10b)*

The raw MVS cloud contains three characteristic noise sources: (1) reconstructed turntable/background plane, (2) scattered photometric outliers at object edges, (3) a floating halo from depth-fusion artifacts. Three filters are applied in sequence:

| Filter | Purpose | Parameters |
|---|---|---|
| Dilated DTU ObsMask | Remove background & out-of-bound points | Dilate 50 mm (50 voxels) at 1 mm resolution |
| Statistical Outlier Removal (SOR) | Remove isolated floating points | k = 20 neighbours, std_ratio = 2.0 |
| Radius Outlier Filter | Remove points with too few local neighbours | min_pts = 5 within radius = 15 mm |

```python
obs_padded  = np.pad(obs_volume_raw, dilate_voxels + 5, constant_values=False)
obs_dilated = binary_dilation(obs_padded, iterations=dilate_voxels)

idx  = np.floor((pts - BB_min) / res).astype(int)
keep = obs_volume[idx[:,0], idx[:,1], idx[:,2]] > 0

pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd, _ = pcd.remove_radius_outlier(nb_points=5, radius=15.0)
```

**Result: 130,029 points** — **36,274 points (21.8%) removed**, all identified as noise, while the object's full surface geometry (cylindrical tin body and brick base) is preserved. The filter cascade is selective rather than aggressive: the object surface is essentially untouched while the turntable plane and outlier scatter disappear.

### 9.8 3D Gaussian Splatting

*(Notebook Section 11)*

#### 9.8.1 Gaussian Representation & Initialization

Each of the cleaned cloud's 130,029 points seeds one 3D Gaussian. Scale is initialized from the mean distance to each point's 3 nearest neighbours (via a `cKDTree`), giving each Gaussian a physically meaningful starting size rather than an arbitrary constant:

```
G(x) = exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ))          Σ = R · diag(s²) · Rᵀ
```

```python
def init_gaussians(pts, cols, scene_extent):
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=4)
    knn_dist = d[:, 1:].mean(axis=1)                     # avg of 3 nearest
    log_scales = np.log(np.clip(knn_dist, 1e-3, scene_extent/10))
    scales = np.tile(log_scales[:, None], (1, 3))        # isotropic init
    quats = np.zeros((n, 4)); quats[:, 0] = 1.0           # identity rotation
    opacities = np.full(n, np.log(0.1 / (1 - 0.1)))       # inverse-sigmoid(0.1)
    return {'means': ..., 'scales': ..., 'quats': ..., 'opacities': ..., 'colors': ...}
```

#### 9.8.2 Differentiable Rendering & Loss

Rendering is alpha compositing of depth-sorted Gaussians, performed by `gsplat.rendering.rasterization`:

```
C = Σᵢ cᵢ αᵢ Π_{j<i} (1 − αⱼ)
```

The training loss combines pixel-level accuracy and structural similarity:

```
L = 0.8 · L₁(Î, I_gt) + 0.2 · L_SSIM(Î, I_gt)
```

Training uses Adam with per-parameter-group learning rates (positions, scales, rotations, opacities, colors), an exponentially decaying position learning rate (`1.6e-4 → 1.6e-6` over 15,000 iterations), and randomly samples one of the 49 ground-truth views per iteration.

#### 9.8.3 Densification Loop (Clone / Split / Prune)

Every 100 iterations, from iteration 500 to 7,500, the mean accumulated 2D position gradient per Gaussian is used to decide:

- **Clone** (small + high-gradient): duplicate — used where a small Gaussian is straining to cover an under-reconstructed region.
- **Split** (large + high-gradient): replace with two smaller children offset by ±scale — used where a large Gaussian spans a region with mixed geometry.
- **Prune**: remove Gaussians with opacity < 0.005 (effectively invisible) or scale > 10% of scene extent (degenerate).
- **Opacity reset** at iteration 3,000: force all opacities back to 0.01, letting the optimizer rediscover which Gaussians are genuinely useful (this produces a visible, expected temporary darkening of renders around that iteration).

```python
avg_grad  = grad_accum / count_accum
big       = max_scale > 0.01 * scene_extent
high_grad = avg_grad > GS_GRAD_THRESH          # 0.0002

clone_mask = high_grad & ~big
split_mask = high_grad & big
prune_mask = (opacity < 0.005) | (scale > 0.1 * scene_extent)

new_lr = LR_INIT * (LR_FINAL / LR_INIT) ** (iter / total_iters)
```

**Training dynamics:** loss falls from ~0.33 to ~0.04; PSNR rises from ~8 dB to ~25 dB (with a noisy phase during active densification); the Gaussian count starts at 130,029, drops sharply as densification prunes redundant/noisy MVS points, and stabilizes around **~70,000 Gaussians** — the pruning phase dominating cloning is expected, since many raw MVS points were noise or redundant.

Rendered previews across training (captured every 1,500 iterations) show the expected trajectory: blurry/overlapping Gaussians at iter 1,500 → temporary darkening at the iter‑3,000 opacity reset → sharp recovery with visible lettering and texture by iter 4,500–6,000 → high-fidelity color, gloss, and fine detail by iter 9,000–15,000. Final side-by-side comparisons across 5 held-out views (1, 13, 25, 37, 49) show the splat reproducing the tin body color, the "OLIVE" lettering, the handprint decoration, and the brick base texture accurately, with minor residual artifacts (a faint background halo, slightly oversaturated color) attributable to the relatively lightweight 15,000-iteration training budget.

### 9.9 Export & Interactive Viewers

*(Notebook Sections 12–13)*

The final Gaussian Splat is exported in a **SuperSplat-compatible `.ply` format** (openable directly at `https://superspl.at/editor`), storing position, normal placeholders, spherical-harmonic DC color coefficients, opacity, log-scale, and quaternion rotation per Gaussian:

```python
def save_gs_ply(path, g):
    means, scales, quats = ...
    opacities = g['opacities'].cpu().numpy().reshape(-1, 1)
    colors    = torch.clamp(g['colors'], 0, 1).cpu().numpy()
    SH_C0 = 0.28209479177387814
    f_dc  = (colors - 0.5) / SH_C0
    # dtype: x,y,z, nx,ny,nz, f_dc_0..2, opacity, scale_0..2, rot_0..3
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))
```

Every stage's point cloud is also given a **self-contained HTML viewer** built with Three.js (loaded via CDN import maps, `OrbitControls` for drag-to-orbit / scroll-to-zoom / right-drag-to-pan), so results can be shared and viewed in any browser with no local setup:

```python
def make_html_viewer(pts, cols, out_path, title='3D Reconstruction', max_pts=300_000):
    # builds a single .html file with an embedded Three.js point-cloud scene
    ...
```

---

## 10. Mathematical Models Reference

| Equation | Name / Role |
|---|---|
| `P = K[R\|t]` | Camera projection: maps a 3D world point to a 2D image pixel |
| `x ∼ PX` (homogeneous) | Projective mapping; perspective division yields pixel coordinates |
| `x'ᵀ F x = 0` | Epipolar constraint — geometric test for correct feature matches |
| `AX = 0`, solved via SVD(A) | DLT triangulation — recovers a 3D point from 2D correspondences |
| `err = ‖x − π(PX)‖₂` | Reprojection error — quality filter, threshold 6.0 px |
| `DoG = [G(kσ) − G(σ)] * I` | SIFT scale-space — detects scale-invariant interest points |
| `NCC(p,d) = Σ[(r−μᵣ)/σᵣ][(n−μₙ)/σₙ]/N` | MVS photo-consistency — illumination-invariant depth scoring |
| `G(x) = exp(−½(x−μ)ᵀΣ⁻¹(x−μ))` | 3D Gaussian — scene primitive for Gaussian Splatting |
| `Σ = R·diag(s²)·Rᵀ` | 3DGS covariance — anisotropic scale + rotation from a quaternion |
| `C = Σ cᵢαᵢ Π(1−αⱼ)` | Alpha compositing — front-to-back color accumulation |
| `L = 0.8·L₁ + 0.2·L_SSIM` | Training loss — joint photometric + structural similarity |

---

## 11. How to Run

1. **Get the dataset.** Obtain the DTU Robot Image Dataset SampleSet (or the specific `3d-reconstruction-dataset` Kaggle dataset referenced in the notebook) and place it so the path structure matches [Section 7](#7-dataset), or update `DATA_ROOT` in the Configuration section.
2. **Provision a GPU environment.** An NVIDIA GPU with CUDA is required for the Gaussian Splatting stage (`gsplat` compiles CUDA kernels on first install, ~3–5 minutes); the classical SfM/MVS stages can technically run CPU-only but will be much slower.
3. **Open `3D_Reconstruction.ipynb`** in Jupyter, Kaggle, Colab, or any compatible notebook environment.
4. **Run all cells top to bottom.** Section 1 installs all dependencies automatically. Sections are self-contained and build on each other's outputs — do not skip sections.
5. **Expect run time** dominated by two stages: MVS plane-sweep densification (many depth-hypothesis evaluations across all reference views) and the 15,000-iteration Gaussian Splatting training loop.
6. **Inspect outputs** in `OUT_DIR` (`/kaggle/working/output` by default, or wherever you configure it) — see [Section 12](#12-output-files).
7. **View results** by opening any of the generated `.html` files directly in a browser (no server needed), or by dragging `gaussian_splat.ply` into [SuperSplat](https://superspl.at/editor), or opening any `.ply` in MeshLab / CloudCompare.

---

## 12. Output Files

| File | Description | Size / Count |
|---|---|---|
| `output/sparse.ply` | Classical SfM sparse cloud | 30,583 pts |
| `output/sparse.html` | Interactive Three.js viewer (sparse) | self-contained |
| `output/dense.ply` | MVS-densified cloud (raw) | 166,303 pts |
| `output/dense.html` | Interactive viewer (raw dense) | self-contained |
| `output/dense_clean.ply` | Cleaned MVS cloud (ObsMask + SOR + radius) | 130,029 pts |
| `output/dense_clean.html` | Interactive viewer (cleaned) | self-contained |
| `output/gaussian_splat.ply` | Final Gaussian Splat (SuperSplat-compatible) | ~70k–130k Gaussians |
| `output/gaussian_splat.html` | Interactive viewer (splat centers) | self-contained |
| `output/viewer.html` | Default viewer (alias of `dense_clean.html`) | — |
| `output/figures/*.png` | All visualization figures (camera poses, SIFT keypoints, match graph, cloud comparisons, training curves, render progression, GT-vs-render) | ~35 MB, 13 figures |
| `output/renders/iter_*.png` | Gaussian Splatting training progression frames (10 checkpoints) | ~6 MB |

---

## 13. Results Summary

| Metric | Value |
|---|---|
| Sparse SfM cloud | 30,583 points |
| Dense MVS cloud | 166,303 points (5.4× sparse) |
| Cleaned dense cloud | 130,029 points (36,274 / 21.8% noise removed) |
| Final Gaussian count | ~70,000 (post-densification, converged) |
| Training iterations | 15,000 |
| Final PSNR | ~25 dB |
| Final training loss (L1 + SSIM) | ~0.04 (down from ~0.33 at start) |
| Best matched view pair | View 009 ↔ View 010 — 1,134 inlier matches |

Qualitatively, the final Gaussian Splat renders accurately reproduce the object's color (dark tin body, red wax-style top, golden rim), the "OLIVE" surface lettering, a handprint decoration, and the brick base texture, with sharp edges and captured specular highlights on the rim. Remaining artifacts (a faint background halo, slightly oversaturated color in places) are consistent with a lightweight 15,000-iteration training run rather than a fundamental pipeline limitation.

---

## 14. Known Issues, Limitations & Design Notes

The configuration cell and code comments document a set of deliberate tuning decisions made after an earlier ("v1") version of the pipeline underperformed. These are worth understanding as design rationale rather than bugs:

- **SfM point yield.** v1 only triangulated points seen in ≥3-view tracks; the current version accepts **two-view tracks** (`MIN_TRACK_LEN = 2`) and triangulates pairwise for every valid match, raising sparse-cloud density substantially (from an estimated ~5–15k points in early testing to the reported 30,583).
- **MVS point yield.** v1 used a larger 11×11 patch, coarser stride (8), and fewer depth hypotheses (64). The current settings (`MVS_PATCH=7`, `MVS_STRIDE=4`, `MVS_DEPTHS=96`) trade compute for a denser, higher-resolution dense cloud.
- **Cleanup necessity.** The raw MVS cloud is not clean enough to feed directly into Gaussian Splatting — the DTU dataset's turntable and background reconstruct as spurious geometry, which is why the three-stage ObsMask/SOR/radius cleanup exists as its own pipeline stage rather than being folded into MVS.
- **Densification was the missing piece in v1** for Gaussian Splatting — without clone/split/prune, opacity reset, and learning-rate scheduling, Gaussian Splatting alone cannot recover sharp geometry from a moderately noisy point initialization. The current pipeline's `GS_ITERS` was raised from 7,000 to 15,000 to give the densification loop enough time to converge.
- **SSIM is simplified.** The notebook uses a global-statistics SSIM approximation (`ssim_simple`) rather than a full windowed/convolutional SSIM, for training speed — this is noted in-code as "sufficient for our loss."
- **Scope of the object.** The dataset is a single controlled studio object (uniform lighting, black background, robot-precise camera calibration). The pipeline's plane-sweep MVS and cleanup thresholds are tuned for this controlled setting and are not validated on uncontrolled, real-world scenes.
- **49 of 64 available views are used**, not the full camera set — a deliberate subsampling choice for speed rather than a dataset limitation.

---

## 15. Future Work

As suggested by the project supervisors, promising directions include:

- **Pure Gaussian Splatting from learned depth priors**, removing the classical SfM/MVS initialization stage entirely.
- **Integration of learning-based depth refinement networks** (e.g., DPT or MiDaS) to improve or replace the plane-sweep MVS stage.
- **Extension to challenging real-world scenes** beyond the controlled DTU laboratory environment — uncontrolled lighting, dynamic backgrounds, and uncalibrated cameras (which would require adding pose estimation back into the SfM stage).

---

## 16. Conclusion

This project implements a complete, end-to-end 3D reconstruction pipeline that combines classical computer vision with modern differentiable rendering, applied to the DTU Robot Image Dataset (`scan1`). Starting from 49 calibrated 2D photographs of a decorative tin can, the pipeline produces a high-quality 3D Gaussian Splatting representation capable of rendering novel views at approximately 25 dB PSNR after 15,000 training iterations.

The classical SfM stage (SIFT, FLANN-RANSAC matching, two-view DLT triangulation) delivers a geometrically consistent sparse cloud of 30,583 points. The plane-sweep MVS stage, scoring 96 depth hypotheses per pixel with illumination-invariant NCC across 4 neighbouring views, densifies this to 166,303 points. The three-stage cleanup pipeline (DTU ObsMask dilation, statistical outlier removal, radius filter) selectively removes 36,274 noise points (21.8%) while preserving the complete object surface at 130,029 points. Finally, the 3D Gaussian Splatting stage — initialized from the MVS cloud and trained with a full densification loop (clone, split, prune) plus an opacity reset — achieves sharp, photorealistic rendering of both fine surface detail and accurate color.

The central takeaway: **classical geometry-based methods and modern differentiable rendering are complementary, not competing.** Classical SfM/MVS provides a reliable, interpretable geometric initialization that gives the Gaussian Splatting optimizer a strong starting point; Gaussian Splatting then refines that geometry into a photorealistic, novel-view-capable representation.

---

## 17. Credits & References

**Team:** Talha Arshad (410179) · Muhammad Taha (417609) · Syed Adnan Aijaz Bukhari (432028)
**Supervisors:** Dr. Shahid Ismail · LE Umair Khalil
**Course:** Computer Vision (CEP), DE‑44, C&SE — College of Electrical & Mechanical Engineering (EME), NUST — May 2026

**Dataset:** DTU Robot Image Dataset (SampleSet / MVS Data), a well-known multi-view stereo benchmark.

**Key techniques and tools referenced:**
- SIFT — Scale-Invariant Feature Transform (Lowe)
- FLANN — Fast Library for Approximate Nearest Neighbors
- RANSAC — Random Sample Consensus
- DLT — Direct Linear Transform triangulation
- 3D Gaussian Splatting — Kerbl et al., 2023, via the `gsplat` library
- SuperSplat — browser-based `.ply` Gaussian Splat viewer/editor (`https://superspl.at/editor`)

**Source documents used to compile this README:**
- `3D_Reconstruction.ipynb` — the full implementation notebook (49 cells, Sections 1–14)
- `3D_Reconstruction_Report.docx` — the formal written course report with figures and derivations
