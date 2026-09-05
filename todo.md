- add register with unigrad, choose sim (for multimodal), convexadam, demons simpleitk, syn, elastix
- add zoom
- correct metrics, atm they are just wrong
- let user easily add own method
-save results
- compare across different runs

ow, 10 ideas for where I'd take this next — grounded in things I actually ran into while working on it, not generic suggestions:

Fix the backend's concurrency bottleneck. I hit this directly while testing: handlers do CPU-bound numpy work inside async def routes, so one heavy request (Hausdorff, SVD) blocks the entire single-worker event loop — nothing else, not even a trivial GET, responds until it's done. Uploading a full set of 3D files at once queued ~25 requests and the server hung for 100+ seconds. Fix: plain def handlers (FastAPI auto-threadpools them) or explicit run_in_threadpool.

Guard against the pathological Dice/Hausdorff case I found. If a real segmentation is compared against a raw (non-segmented) image — which happens transiently if you upload Fixed/Moving segmentations one at a time rather than together — _binarize treats nearly every voxel as foreground, and Hausdorff's cdist blows up. Worth validating segmentation-shaped input before computing, or refusing with a clear message instead of silently grinding.

Use real voxel spacing. The Jacobian gradients and Hausdorff distances all assume isotropic 1.0 spacing, but nibabel already gives you the affine — for anisotropic real scans (NLST, ACDC) the numbers are voxel counts dressed up as if they were physical distances. Reading spacing from the affine would make every metric mm-accurate.

Linked crosshair across tiles. Click a point in Fixed, see the corresponding point highlighted in Moving/Warped. This is the single biggest QA-workflow gap right now — you can eyeball similarity but can't confirm a specific anatomical point actually landed where it should.

Blink/checkerboard comparison mode. The static Difference Image is useful, but flicker-blending Fixed↔Warped (or a checkerboard split) is the standard radiology trick for spotting misregistration the naked eye glosses over in a diff image.

Surface the registration-run history. The backend already keeps every run under registration_jobs/ — the UI just never shows you it exists. A small "past runs" list would let you reload and compare an earlier affine attempt against a later deformable one instead of only ever seeing the latest.

Two-run side-by-side comparison. Related to #6: run affine and deformable back to back and diff their metrics directly, rather than the second run silently overwriting the first's warped/metrics state.

Named, editable structure labels. Segmentations show up as generic "Label 1" / "Label 2" with auto-generated colors. Letting a user type "Lung" / "Tumor" and pick a color would make the per-structure Dice/HD breakdown much more legible for anything beyond a synthetic demo.

Export the results. Right now everything lives only in the browser tab — no way to download the warped image, displacement field, or a metrics CSV/JSON, let alone a shareable snapshot of the current view.

Keyboard/wheel slicing. Scrubbing through a volume currently means mouse-dragging the slider precisely. Arrow keys or scroll-over-viewer for single-slice steps would make 3D exploration much less fiddly.