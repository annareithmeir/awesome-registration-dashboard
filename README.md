![Awesome Image Registration Dashboard](app-screenshot.png)

# Awesome Image Registration Dashboard

Load two scans, register them, and see exactly how well it worked — down to the last voxel.

A local web app for running and inspecting medical image registration: load a fixed/moving pair (2D or 3D NIfTI), run affine or deformable registration in the browser, and drill into the result with per-structure overlap metrics, Jacobian/topology diagnostics, and a displacement-field visualizer — no notebook required.

## Features

- **Load anything, or start from a sample.** Upload fixed/moving images and segmentations (`.nii` / `.nii.gz`), or pick a bundled sample set to explore the app immediately — a synthetic 2D expand/shrink pair, plus optional real 2D/3D cardiac MRI and lung CT sets you can generate locally (see [Sample data](#sample-data)).
- **Run registration in-browser.** Affine (`reg_aladin`) or deformable (`reg_f3d`) registration via [NiftyReg](https://github.com/KCL-BMEIS/niftyreg), with tunable iterations, regularization weight, and control-point spacing — or load an already-registered warped image + displacement field from elsewhere.
- **Full 3D navigation.** Scrub through slices with the floating slider or the mouse wheel, switch between sagittal/coronal/axial planes, and rotate the current view 90° — all in sync across every open viewer.
- **Segmentation overlay.** Toggle any image between raw intensities and its segmentation, with an adjustable overlay opacity.
- **Registration-quality metrics, not just a warped image:**
  - Dice, Dice95, Hausdorff, and Hausdorff95 — overall and per structure, before vs. after registration.
  - Jacobian determinant, with automatic folding/topology-preservation detection.
  - Log-Jacobian (symmetric growth/shrinkage), Shear Index, and Inverse Consistency Error, each with its own heatmap.
- **Displacement field visualization** — warped grid overlay and a hue/brightness color wheel encoding direction and magnitude.
- **Difference image** between fixed vs. moving or fixed vs. warped, to spot misregistration at a glance.

## Setup

Requires Python 3.12+ and Node 18+.

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Registration additionally requires the [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) command-line tools (`reg_aladin`, `reg_f3d`, `reg_resample`, `reg_transform`) — `pip install niftyreg` builds them for you, or install NiftyReg separately and make sure it's on `PATH`. Everything else (loading images, viewing metrics) works without it.

### Frontend

```bash
cd frontend
npm install
```

## Run

Start the backend:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Start the frontend, in a separate terminal:

```bash
cd frontend
npm run dev
```

Open the local Vite URL shown in the terminal.

## Sample data

A small synthetic 2D pair (two circles, one expanding, one shrinking) ships with the app and can be generated anytime from the UI's "Load sample data" panel, or directly with:

```bash
python -c "from backend.app import test_data; test_data.generate_and_save_samples()"
```

`backend/app/test_data.py` also has `generate_acdc_sample()` and `generate_nlst_sample()`, which build a cropped fixed/moving pair from a local copy of the [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) cardiac MRI or [Learn2Reg NLST](https://learn2reg.grand-challenge.org/) lung CT datasets. These require your own local copy of the source data (point `acdc_root` / `nlst_root` at it) — the datasets themselves aren't part of this repo and are governed by their own usage terms.

## Tests

```bash
cd backend
pytest
```

## Notes

- `.pt` displacement files are raw tensors saved with `torch.save()` — no metadata. Supported layouts: channel-first (`CxHxW`, `CxDxHxW`) or channel-last (`HxWxC`, `DxHxWxC`); 2D or 3D.
- The frontend talks to the backend at `http://localhost:8000`.
- The backend processes requests synchronously, so it's built for one interactive session at a time rather than concurrent/production use.

## Acknowledgments

- [NiftyReg](https://github.com/KCL-BMEIS/niftyreg) for the registration engine.
- [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/) and [Learn2Reg](https://learn2reg.grand-challenge.org/) for the real-data sample generators.
