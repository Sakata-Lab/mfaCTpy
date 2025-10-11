# uCT2CCF: MicroCT Mouse Brain Registration to Allen CCF

## User Guide v1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Structure](#data-structure)
4. [Workflow Overview](#workflow-overview)
5. [Step-by-Step Instructions](#step-by-step-instructions)
6. [Troubleshooting](#troubleshooting)
7. [Tips for Best Results](#tips-for-best-results)

---

## Overview

This package provides a complete workflow for registering microCT-scanned mouse brain images to the Allen Common Coordinate Framework (CCF). The workflow enables:

- **Automated fiber tracking** for implanted optical fibers
- **Brain region identification** using Allen CCF annotations
- **3D visualization** of fiber locations in anatomical context

### Key Features

- Interactive midline alignment with axis verification
- Landmark-based registration with optional intensity-based refinement
- Manual fiber tracking with automatic region identification
- 3D visualization with slice planes
- Movie generation for presentations

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux

### Required Python Packages

```bash
pip install numpy tifffile matplotlib scipy SimpleITK pandas nrrd requests
```

### Optional Packages

For 3D visualization:
```bash
pip install mpl_toolkits
```

### Project Structure

Create the following folder structure:

```
C:\DATA\MFA\uCT\uCT2CCF\
├── data\
│   ├── raw\                    # Place DICOM files here
│   ├── processed\              # Processed images
│   └── ccf\                    # Allen CCF data
├── outputs\                    # Analysis results
└── src\                        # Python scripts
```

---

## Data Structure

### Input Data

- **MicroCT DICOM files**: Raw scanner output
- **OR pre-converted .tif**: 3D TIFF stack

### Expected Image Properties

- **Resolution**: ~20 μm isotropic (typical for microCT)
- **Content**: Mouse head with skull, brain, and implanted fibers
- **Fiber diameter**: 50 μm (appears as 2-3 pixels at 20 μm resolution)

### Allen CCF Data

The Allen CCF atlas (25 μm resolution) will be automatically downloaded:
- **Annotation volume**: Brain region labels
- **Template volume**: MRI-like reference image
- **Ontology JSON**: Region hierarchy and names

---

## Workflow Overview

```
1. DICOM → TIFF Conversion
   ↓
2. Midline Alignment (Interactive)
   ↓
3. Axis Verification (Optional)
   ↓
4. Landmark Registration (Interactive)
   ↓
5. Fiber Tracking (Interactive)
   ↓
6. 3D Visualization & Reports
```

**Total time**: 2-4 hours (depending on number of landmarks/fibers)

---

## Step-by-Step Instructions

### Step 1: Convert DICOM to TIFF

**Script**: `dicom_loader.py`

**Purpose**: Convert DICOM series to a single 3D TIFF file

**Instructions**:

1. Run the script:
   ```bash
   python dicom_loader.py
   ```

2. When prompted, select the folder containing DICOM files

3. Choose save location for output TIFF

4. The script will:
   - Load all DICOM slices
   - Combine into 3D volume
   - Normalize to 16-bit
   - Save as multi-page TIFF

**Output**: `microct_original.tif` in `data/` folder

**Time**: 2-5 minutes

---

### Step 2: Midline Alignment

**Script**: `midline_alignment.py`

**Purpose**: Align brain to midline and correct orientation

#### 2a. View Selection

When you run the script, you'll first see three orthogonal views of your data. Click on the view that shows horizontal (top-down) brain sections - this is typically the best view for marking the brain midline.

#### 2b. Mark Midline Points

**Instructions**:

1. **Navigate slices**: Use the slider or mouse wheel

2. **Mark points**: LEFT-CLICK along the brain midline
   - Mark 2-3 points per slice
   - Use 5-10 slices spanning the brain
   - Points should follow the sagittal midline

3. **Switch views** (if needed): Use view buttons to mark in multiple planes

4. **Verify coverage**: Ensure points span anterior-posterior extent

5. **Clear if needed**:
   - "Clear Slice": Remove points from current slice
   - "Clear All": Start over

6. **Finish**: Click "Done" when satisfied

#### 2c. Apply Alignment

The script will:
- Fit a plane to your midline points
- Calculate rotation needed to align midline vertically
- Apply rotation at full resolution (**takes 5-30 minutes**)
- Show before/after visualization

**Check the result**: The green dashed line should align with the brain midline in the visualization.

#### 2d. Axis Verification (Recommended)

After alignment, you can verify that axes are correctly labeled. The expected orientation should show the brain from front/back in coronal view, from top in axial view (with midline vertical), and from the side in sagittal view.

If views don't match:
- Use **Swap** buttons to exchange axes
- Use **Flip** buttons to mirror along an axis

**Output**: `microct_aligned.tif` (and optionally `microct_aligned_corrected.tif`)

**Time**: 15-30 minutes (marking) + 5-30 minutes (processing)

---

### Step 3: Landmark Registration

**Script**: `landmark_registration.py`

**Purpose**: Register aligned microCT to Allen CCF atlas

#### 3a. Landmark Selection

**Good landmarks for mouse brain**:
- **Bregma**: Intersection of coronal and sagittal sutures
- **Lambda**: Posterior end of sagittal suture
- **Brain tips**: Anterior, posterior, ventral
- **Brain edges**: Left and right at widest point
- **Ventricle corners** (if visible)
- **Olfactory bulb tips**

**Instructions**:

1. **Click on MOVING image** (microCT) first
2. **Navigate** to find the landmark clearly
3. **Click corresponding point** on FIXED image (Allen CCF)
4. **Switch views** (Z/Y/X buttons) to find landmarks in different planes
5. **Select 6-12 landmark pairs** (minimum 6, more is better)

**Tips**:
- Use multiple anatomical views to increase accuracy
- Distribute landmarks throughout the brain volume
- Prioritize clear, unambiguous features
- Check the landmark list occasionally ("List All" button)

#### 3b. Registration

The script will:
- Compute affine transformation from landmarks
- Apply transformation to align images
- Show registration quality metrics
- Generate visualization

**Quality metrics**:
- **Mean error**: < 0.5 mm is excellent, < 1.0 mm is good
- **Max error**: < 2.0 mm is acceptable

#### 3c. Optional Refinement

You'll be asked: "Refine registration with intensity-based optimization?"

- **Yes**: Improves alignment using image intensities (recommended, takes 10-20 min)
- **No**: Use landmark-based result only

**Output**: 
- `microct_registered.tif`: Registered image
- `transform_landmark.tfm`: Transform file
- `registration_metrics.json`: Quality metrics
- Visualization images

**Time**: 30-60 minutes (landmark selection) + 5-20 minutes (computation)

---

### Step 4: Fiber Tracking

**Script**: `fiber_tracker.py`

**Purpose**: Track optical fibers and identify brain regions

#### 4a. Setup

The script will check for required files and ask whether you want to use the registered image (recommended) or aligned image with transform. Using the registered image provides direct coordinate mapping and is simpler and more accurate.

The script will also check for Allen CCF annotation and download if needed.

#### 4b. Previous Work

If previous tracking work is found, you can choose to continue (load previous fibers and add more), replace (start fresh), or quit.

#### 4c. Track Fibers

**Instructions**:

1. **Navigate** to fiber entry point at brain surface

2. **Click** to mark **FIBER TOP** (entry point)

3. **Navigate** to fiber tip deep in brain

4. **Click** to mark **FIBER BOTTOM** (tip)

5. The script automatically identifies the brain region at the fiber tip

6. **Click "Next Fiber"** to save and start the next fiber

7. **Repeat** for all fibers

**Shortcuts**:
- `u`: Undo current point
- `Shift+U`: Undo last saved fiber
- `n`: Next fiber (save current)
- `s`: Save progress
- `Enter`: Finish

**Region identification**:
The script displays region name, acronym, CCF coordinates, and hierarchical path for each fiber tip.

#### 4d. Review Outputs

**Generated files**:
- `fiber_data.json`: Complete fiber information
- `fiber_report.csv`: Spreadsheet-compatible data
- `fiber_summary.txt`: Human-readable summary
- `fiber_horizontal_view.png`: Coronal overview
- `fiber_ccf_overlay.png`: Fibers on CCF annotation

**Output**: Complete fiber tracking data in `outputs/` folder

**Time**: 5-10 minutes per fiber

---

### Step 5: 3D Visualization

**Script**: `fiber_visualizer_3d.py`

**Purpose**: Interactive 3D visualization of tracked fibers

**Features**:
- Rotate 3D view with mouse
- Zoom with mouse wheel
- Toggle slice planes (coronal, sagittal, axial)
- Show/hide individual fibers
- Switch between microCT and Allen CCF
- Preset camera angles

**Instructions**:

1. Run the script:
   ```bash
   python fiber_visualizer_3d.py
   ```

2. **Interact**:
   - **Drag**: Rotate view
   - **Scroll**: Zoom in/out
   - **Sliders**: Navigate slice positions
   - **Checkboxes**: Toggle fiber visibility
   - **Buttons**: Change views, reset zoom

**Time**: Interactive (no time limit)

---

### Step 6: Create Movies (Optional)

**Script**: `movie_creator.py`

**Purpose**: Generate movies showing brain slices for presentations

**Instructions**:

1. Run the script:
   ```bash
   python movie_creator.py
   ```

2. Select input TIFF file

3. In the GUI:
   - Select axis/axes for movies
   - Name each axis (e.g., "coronal", "sagittal")
   - Set frame rate (default: 20 fps)
   - Choose compression quality
   - Optionally flip orientation

4. Click "Create Movies"

**Output**: MP4 movies showing slices through selected axes

**Time**: 2-5 minutes per movie

---

## Troubleshooting

### Issue: Midline alignment looks wrong

**Symptoms**: Green line doesn't align with midline after rotation

**Solutions**:
1. Check if you marked enough points (need 2-3 points on 5-10 slices)
2. Ensure points truly follow the midline (not off-center)
3. Try marking in a different view (axial vs coronal)
4. If rotation is in the wrong direction, set `reverse=True` in the `apply_alignment()` call

### Issue: Axes appear swapped after alignment

**Symptoms**: Coronal view shows top-down, or other axis confusion

**Solutions**:
1. Use the axis verification step and swap axes as needed using the Swap buttons
2. Save the corrected image for subsequent steps
3. Document which swaps you used for reproducibility

### Issue: Registration error is high (> 1 mm)

**Symptoms**: Mean landmark error > 1.0 mm

**Solutions**:
1. Review landmark pairs - look for outliers in the error list
2. Remove problematic landmarks and re-run
3. Add more landmarks in poorly-aligned regions
4. Ensure landmarks are clear anatomical features visible in both images
5. Try intensity-based refinement

### Issue: Fiber regions identified incorrectly

**Symptoms**: Region name doesn't match expected anatomy

**Solutions**:
1. Check if you're using the registered image (recommended) vs aligned image with transform
2. Verify CCF annotation loaded correctly
3. Check registration quality - poor registration → wrong regions
4. Verify fiber tip is actually at the intended location
5. Review the CCF coordinates printed during tracking

### Issue: Scripts run very slowly

**Symptoms**: Processing takes much longer than expected times

**Solutions**:
1. Check available RAM (need ~8-16 GB for typical datasets)
2. Close other applications
3. For midline alignment, full resolution processing is slow (5-30 min is normal)
4. Consider using a machine with more CPU cores

### Issue: Allen CCF download fails

**Symptoms**: Cannot download annotation or template

**Solutions**:
1. Check internet connection
2. Try manual download from Allen Institute:
   - URL: http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
3. Place files in `data/ccf/` folder
4. File names: `annotation_25.nrrd`, `average_template_25.nrrd`, `structure_tree.json`

---

## Tips for Best Results

### Imaging Tips

1. **Scan at highest resolution possible** (20 μm or better)
2. **Include clear skull landmarks** (bregma, lambda) if visible
3. **Ensure good contrast** between brain tissue and background
4. **Center the brain** in the scan volume

### Midline Alignment Tips

1. **Mark points carefully** - accuracy here affects all downstream analyses
2. **Use multiple slices** spanning anterior-posterior extent
3. **Verify alignment** before proceeding to registration
4. **Save midline points** so you can reload if needed

### Registration Tips

1. **Select 8-12 landmarks** for best results (minimum 6)
2. **Distribute landmarks** throughout the volume (don't cluster)
3. **Use clear, unambiguous features** that are visible in both images
4. **Check different anatomical planes** to find best landmark views
5. **Always use intensity-based refinement** for best accuracy

### Fiber Tracking Tips

1. Fibers appear as bright thin white lines in the microCT images. Since fibers are penetrated from the surface to deep brain structures, they appear as tiny white dots in horizontal slices.
2. **Navigate through multiple slices** to confirm fiber path
3. **Mark entry point** at brain surface (not in skull)
4. **Mark tip accurately** - small errors change region identification
5. **Save progress frequently** using the "Save Progress" button
6. **Use Undo** liberally if you make a mistake

### Visualization Tips

1. **Start with 3D view** to get overall orientation
2. **Toggle slice planes** to understand fiber locations in context
3. **Hide fibers individually** to reduce visual clutter
4. **Use preset views** (Top/Side/Front) for standard orientations
5. **Export views** by taking screenshots for presentations

---

## File Format Reference

### Input Files

| File | Format | Description |
|------|--------|-------------|
| DICOM series | `.dcm` | Raw microCT scan |
| MicroCT volume | `.tif` | 3D TIFF stack (Z, Y, X) |

### Intermediate Files

| File | Format | Description |
|------|--------|-------------|
| `microct_aligned.tif` | `.tif` | Midline-aligned image |
| `microct_registered.tif` | `.tif` | CCF-registered image |
| `transform_landmark.tfm` | `.tfm` | SimpleITK transform |
| `landmarks.json` | `.json` | Registration landmarks |
| `midline_points.json` | `.json` | Midline alignment points |

### Output Files

| File | Format | Description |
|------|--------|-------------|
| `fiber_data.json` | `.json` | Complete fiber data |
| `fiber_report.csv` | `.csv` | Spreadsheet-compatible |
| `fiber_summary.txt` | `.txt` | Human-readable report |
| `fiber_horizontal_view.png` | `.png` | Coronal overview |
| `fiber_ccf_overlay.png` | `.png` | CCF overlay visualization |
| `*_movie.mp4` | `.mp4` | Slice movies |

---

## Coordinate Conventions

### Image Array Convention

All 3D arrays use **(Z, Y, X)** convention:
- **Z**: Dorsal-Ventral (top-bottom)
- **Y**: Anterior-Posterior (front-back)
- **X**: Medial-Lateral (left-right)

### Physical Space Convention

SimpleITK and Allen CCF use **(x, y, z)** convention:
- **x**: Left-Right
- **y**: Anterior-Posterior
- **z**: Dorsal-Ventral

### Slice Views

- **Coronal**: Slicing through Z, viewing Y-X plane (front/back view)
- **Axial**: Slicing through Y, viewing Z-X plane (top/down view)
- **Sagittal**: Slicing through X, viewing Z-Y plane (side view)

---

## Citation

If you use this package in your research, please cite:

```
Allen Institute for Brain Science (2011). Allen Mouse Brain Atlas.
Available from: http://mouse.brain-map.org/

Wang et al. (2020). The Allen Mouse Brain Common Coordinate Framework:
A 3D Reference Atlas. Cell, 181(4), 936-953.
```

---

## Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Tips for Best Results](#tips-for-best-results)
3. Verify all file paths are correct in your scripts
4. Check that all dependencies are installed

---

## Version History

- **v1.0** (2025): Initial release
  - DICOM to TIFF conversion
  - Interactive midline alignment with axis verification
  - Landmark-based registration
  - Fiber tracking with CCF integration
  - 3D visualization
  - Movie generation

---

**Good luck with your analysis!**