"""Background registration jobs with time-estimated progress.

Four methods, in two architectural families:

- affine (NiftyReg reg_aladin) and deformable (NiftyReg reg_f3d) shell out
  to the NiftyReg CLI tools as a subprocess, same as before.
- syn (ANTs, via antspyx) and demons (SimpleITK) run in-process instead -
  there's no CLI to shell out to, and both libraries are heavy optional
  dependencies, so they're imported lazily inside run_syn_registration /
  run_demons_registration rather than at module load time.

Displacement field channel-order convention: this app's `.pt` files use
component i <-> spatial axis (N-1-i) - e.g. for a 2D field, channel 0 is the
X/column displacement and channel 1 is Y/row (that's what NiftyReg's
reg_transform already outputs, and what main.py's preview/Jacobian code
assumes). ANTs' own displacement fields use the *opposite* convention
(component i <-> spatial axis i directly), verified empirically with a
hand-constructed known transform - see run_syn_registration for the fix.
SimpleITK's fields already match this app's convention directly, verified
the same way - see run_demons_registration.
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid

import nibabel as nib
import numpy as np

try:
    import torch
except Exception:
    torch = None


DEFAULT_BENDING_ENERGY = 0.005

# NiftyReg's own defaults (reg_aladin -maxit 5, reg_f3d -maxit 150) converge
# fast on easy synthetic data but cut real registrations (e.g. cardiac MRI)
# off well before they've actually converged. Run with 5x the iteration
# budget per level so the optimizer has room to finish properly instead of
# stopping early.
ALADIN_MAXIT = 5 * 5
F3D_MAXIT = 150 * 5

# SyN's iteration count is split across a 3-level (coarse-to-fine) schedule;
# this is the finest level's count, with coarser levels getting proportionally
# fewer. Demons has no multi-resolution schedule - it's a flat iteration
# count - and converges fast since it's a purely local optimization, hence
# the much lower default than the NiftyReg/SyN methods above.
SYN_DEFAULT_ITERATIONS = 100
DEMONS_DEFAULT_ITERATIONS = 50
# Standard deviation (voxels) of the Gaussian used to smooth the deformation
# field each iteration - Demons' only regularization knob, playing the same
# role reg_f3d's bending energy does (higher = smoother/stiffer).
DEMONS_DEFAULT_SMOOTHING = 1.5

# Per-phase metadata. `weight` is the fraction of overall progress this phase
# accounts for (within whichever phase set actually runs); `expected` is a
# rough duration in seconds used purely to animate the progress bar between
# phase boundaries (none of these tools expose a numeric completion
# percentage). syn/demons run in-process rather than as a subprocess and are
# dramatically faster than the NiftyReg CLI tools on comparable data (real
# measurement, not a guess: ~0.5s vs ~70s for a similarly-sized 3D volume).
PHASE_DEFS = {
    "affine": {"label": "Running affine registration (reg_aladin)…", "weight": 0.7, "expected": 25.0},
    "deformable": {"label": "Running deformable registration (reg_f3d)…", "weight": 0.75, "expected": 70.0},
    "syn": {"label": "Running SyN registration (ANTs)…", "weight": 0.75, "expected": 3.0},
    "demons": {"label": "Running Demons registration (SimpleITK)…", "weight": 0.6, "expected": 1.5},
    "field": {"label": "Computing displacement field…", "weight": 0.08, "expected": 1.5},
    "segmentation": {"label": "Resampling segmentation…", "weight": 0.17, "expected": 2.5},
}

REGISTRATION_TYPES = ("affine", "deformable", "syn", "demons")

_JOBS = {}
_JOBS_LOCK = threading.Lock()


class RegistrationError(RuntimeError):
    pass


def _phase_keys(registration_type, has_segmentation):
    keys = [registration_type, "field"]
    if has_segmentation:
        keys.append("segmentation")
    return keys


def _find_tool(name):
    path = shutil.which(name)
    if path:
        return path
    # The `niftyreg` PyPI package compiles the NiftyReg CLI tools from source
    # and ships them under its own package directory rather than on PATH.
    try:
        import niftyreg
        candidate = os.path.join(niftyreg.bin_path, name)
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    raise RegistrationError(
        f"NiftyReg tool '{name}' was not found on PATH. Install NiftyReg (reg_aladin, reg_f3d, "
        "reg_resample, reg_transform) and make sure it is on PATH, or `pip install niftyreg`."
    )


def _run(cmd, log_path):
    with open(log_path, "a") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RegistrationError(f"Command failed ({result.returncode}): {' '.join(cmd)}. See {log_path}")


def create_job(
    jobs_dir,
    registration_type="deformable",
    bending_energy=None,
    max_iterations=None,
    grid_spacing=None,
    smoothing=None,
):
    if registration_type not in REGISTRATION_TYPES:
        raise RegistrationError(f"registration_type must be one of: {', '.join(REGISTRATION_TYPES)}")
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(jobs_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    now = time.time()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "pending",
            "phase_key": None,
            "phase_start": now,
            "created_at": now,
            "has_segmentation": False,
            "registration_type": registration_type,
            "bending_energy": bending_energy,
            "max_iterations": max_iterations,
            "grid_spacing": grid_spacing,
            "smoothing": smoothing,
            "error": None,
            "results": None,
            "job_dir": job_dir,
        }
    return job_id, job_dir


def _set(job_id, **fields):
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(fields)


def _get(job_id):
    with _JOBS_LOCK:
        return dict(_JOBS[job_id])


def _enter_phase(job_id, key):
    _set(job_id, phase_key=key, phase_start=time.time(), status="running")


def get_status(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        job = dict(job)

    keys = _phase_keys(job["registration_type"], job["has_segmentation"])
    phases = [{"key": key, **PHASE_DEFS[key]} for key in keys]
    total_weight = sum(p["weight"] for p in phases)

    if job["status"] == "done":
        progress = 1.0
    elif job["status"] == "error":
        progress = None
    else:
        completed_weight = 0.0
        current = None
        for phase in phases:
            if phase["key"] == job["phase_key"]:
                current = phase
                break
            completed_weight += phase["weight"]
        if current is None:
            current = phases[0]
        elapsed = max(0.0, time.time() - job["phase_start"])
        frac = min(1.0, elapsed / current["expected"]) if current["expected"] > 0 else 1.0
        progress = min(0.97, (completed_weight + frac * current["weight"]) / total_weight)

    if job["status"] == "done":
        phase_label = "Done"
    elif job["status"] == "pending":
        phase_label = "Starting…"
    elif job["status"] == "running":
        phase_label = PHASE_DEFS.get(job["phase_key"], {}).get("label", "Running…")
    else:
        phase_label = None

    return {
        "status": job["status"],
        "phase": phase_label,
        "progress": progress,
        "error": job["error"],
        "results": job["results"],
    }


def _to_pt_displacement(nifti_path, pt_path):
    img = nib.load(nifti_path)
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    arr = np.squeeze(arr)
    # NiftyReg saves deformation/displacement fields as (X, Y, Z, 1, 3) (or
    # (X, Y, 1, 1, 2) in 2D) NIfTI volumes; squeeze leaves the channel axis last.
    if arr.ndim not in (3, 4):
        raise RegistrationError(f"Unexpected displacement field shape {arr.shape}")
    if torch is None:
        raise RegistrationError("PyTorch is required to save the displacement field")
    torch.save(torch.from_numpy(arr.astype(np.float32)), pt_path)


def _run_niftyreg_pipeline(job_id, job, fixed_path, moving_path, moving_seg_path, job_dir, log_path):
    registration_type = job["registration_type"]
    bending_energy = job["bending_energy"]
    grid_spacing = job.get("grid_spacing")

    reg_transform = _find_tool("reg_transform")
    reg_resample = _find_tool("reg_resample") if moving_seg_path else None

    warped_path = os.path.join(job_dir, "warped.nii.gz")

    if registration_type == "affine":
        reg_aladin = _find_tool("reg_aladin")
        transform_path = os.path.join(job_dir, "affine.txt")
        maxit = job.get("max_iterations") or ALADIN_MAXIT
        _enter_phase(job_id, "affine")
        _run(
            [
                reg_aladin, "-ref", fixed_path, "-flo", moving_path,
                "-aff", transform_path, "-res", warped_path,
                "-maxit", str(int(maxit)),
            ],
            log_path,
        )
    else:
        reg_f3d = _find_tool("reg_f3d")
        transform_path = os.path.join(job_dir, "cpp.nii.gz")
        maxit = job.get("max_iterations") or F3D_MAXIT
        _enter_phase(job_id, "deformable")
        cmd = [
            reg_f3d, "-ref", fixed_path, "-flo", moving_path,
            "-cpp", transform_path, "-res", warped_path,
            "-maxit", str(int(maxit)),
        ]
        if bending_energy is not None:
            cmd += ["-be", str(bending_energy)]
        if grid_spacing is not None:
            # Negative value = spacing in voxels rather than mm (see reg_f3d -sx/-sy).
            cmd += ["-sx", str(-abs(float(grid_spacing))), "-sy", str(-abs(float(grid_spacing)))]
        _run(cmd, log_path)

    disp_nifti = os.path.join(job_dir, "displacement.nii.gz")
    disp_pt = os.path.join(job_dir, "displacement.pt")
    _enter_phase(job_id, "field")
    _run([reg_transform, "-ref", fixed_path, "-disp", transform_path, disp_nifti], log_path)
    _to_pt_displacement(disp_nifti, disp_pt)

    warped_seg_path = None
    if moving_seg_path:
        warped_seg_path = os.path.join(job_dir, "warped_seg.nii.gz")
        _enter_phase(job_id, "segmentation")
        _run(
            [reg_resample, "-ref", fixed_path, "-flo", moving_seg_path, "-trans", transform_path, "-res", warped_seg_path, "-inter", "0"],
            log_path,
        )

    return warped_path, disp_pt, warped_seg_path


def _load_array(path):
    img = nib.load(path)
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    return np.squeeze(arr)


def _save_nifti(arr, path, dtype=np.float32):
    nib.save(nib.Nifti1Image(np.asarray(arr).astype(dtype), np.eye(4)), path)


def run_syn_registration(fixed_arr, moving_arr, moving_seg_arr=None, max_iterations=None):
    """ANTs SyN registration, run in-process (no subprocess/CLI). Returns
    (warped_arr, displacement_arr, warped_seg_arr) - warped_seg_arr is None
    when moving_seg_arr wasn't given. displacement_arr is already converted
    to this app's channel-order convention (see the module docstring).

    A pure function (numpy arrays in, numpy arrays out) deliberately kept
    separate from job/phase bookkeeping, so it's directly unit-testable.
    """
    import ants  # deferred: antspyx is a heavy optional dependency

    fimg = ants.from_numpy(np.asarray(fixed_arr, dtype=np.float32))
    mimg = ants.from_numpy(np.asarray(moving_arr, dtype=np.float32))

    max_iterations = int(max_iterations or SYN_DEFAULT_ITERATIONS)
    # A standard coarse-to-fine schedule: full iteration budget at the
    # finest level, proportionally less at coarser levels.
    reg_iterations = (max_iterations, max(1, max_iterations // 2), max(1, max_iterations // 4))

    out = ants.registration(fixed=fimg, moving=mimg, type_of_transform="SyN", reg_iterations=reg_iterations)
    warped_arr = out["warpedmovout"].numpy().astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        composed_path = ants.apply_transforms(
            fixed=fimg, moving=mimg, transformlist=out["fwdtransforms"], compose=os.path.join(tmp, "field_"),
        )
        ants_field = ants.image_read(composed_path).numpy()
    # ANTs' field component order is (spatial axis 0, 1, ...) directly; this
    # app's convention is the reverse (component i <-> spatial axis N-1-i) -
    # see the module docstring.
    disp_arr = ants_field[..., ::-1].astype(np.float32)

    warped_seg_arr = None
    if moving_seg_arr is not None:
        seg_img = ants.from_numpy(np.asarray(moving_seg_arr, dtype=np.float32))
        warped_seg_img = ants.apply_transforms(
            fixed=fimg, moving=seg_img, transformlist=out["fwdtransforms"], interpolator="nearestNeighbor",
        )
        warped_seg_arr = np.rint(warped_seg_img.numpy()).astype(np.uint8)

    return warped_arr, disp_arr, warped_seg_arr


def run_demons_registration(fixed_arr, moving_arr, moving_seg_arr=None, iterations=None, smoothing=None):
    """SimpleITK (fast symmetric forces) Demons registration, run
    in-process. Returns (warped_arr, displacement_arr, warped_seg_arr), same
    shape as run_syn_registration above.

    Demons is a purely local, gradient-based method - unlike SyN/NiftyReg it
    has no built-in global (affine) search step, so it assumes fixed/moving
    are already roughly aligned; run an affine registration first if they
    aren't.
    """
    import SimpleITK as sitk  # deferred: only needed for this method

    fixed_img = sitk.GetImageFromArray(np.asarray(fixed_arr, dtype=np.float32))
    moving_img = sitk.GetImageFromArray(np.asarray(moving_arr, dtype=np.float32))
    # Demons compares intensities directly (no mutual-information-style
    # metric), so it assumes fixed/moving share a similar intensity
    # distribution - histogram-match moving onto fixed first.
    matched_moving = sitk.HistogramMatching(
        moving_img, fixed_img, numberOfHistogramLevels=1024, numberOfMatchPoints=7, thresholdAtMeanIntensity=True,
    )

    demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons_filter.SetNumberOfIterations(int(iterations or DEMONS_DEFAULT_ITERATIONS))
    demons_filter.SetStandardDeviations(float(smoothing if smoothing is not None else DEMONS_DEFAULT_SMOOTHING))
    field_img = demons_filter.Execute(fixed_img, matched_moving)

    # Extract the field's array *before* constructing a transform from it -
    # DisplacementFieldTransform takes ownership of the image's buffer,
    # which leaves field_img emptied out from under us afterward.
    # SimpleITK's own component order already matches this app's convention
    # directly (unlike ANTs above) - see the module docstring.
    disp_arr = sitk.GetArrayFromImage(field_img).astype(np.float32)

    disp_tx = sitk.DisplacementFieldTransform(field_img)
    warped_img = sitk.Resample(moving_img, fixed_img, disp_tx, sitk.sitkLinear, 0.0, moving_img.GetPixelID())
    warped_arr = sitk.GetArrayFromImage(warped_img).astype(np.float32)

    warped_seg_arr = None
    if moving_seg_arr is not None:
        seg_img = sitk.GetImageFromArray(np.asarray(moving_seg_arr, dtype=np.float32))
        warped_seg_img = sitk.Resample(seg_img, fixed_img, disp_tx, sitk.sitkNearestNeighbor, 0.0, seg_img.GetPixelID())
        warped_seg_arr = np.rint(sitk.GetArrayFromImage(warped_seg_img)).astype(np.uint8)

    return warped_arr, disp_arr, warped_seg_arr


def _run_inprocess_pipeline(job_id, job, fixed_path, moving_path, moving_seg_path, job_dir, phase_key, run_fn):
    _enter_phase(job_id, phase_key)
    fixed_arr = _load_array(fixed_path)
    moving_arr = _load_array(moving_path)
    moving_seg_arr = _load_array(moving_seg_path) if moving_seg_path else None

    warped_arr, disp_arr, warped_seg_arr = run_fn(fixed_arr, moving_arr, moving_seg_arr)

    _enter_phase(job_id, "field")
    warped_path = os.path.join(job_dir, "warped.nii.gz")
    _save_nifti(warped_arr, warped_path)

    if torch is None:
        raise RegistrationError("PyTorch is required to save the displacement field")
    disp_pt = os.path.join(job_dir, "displacement.pt")
    torch.save(torch.from_numpy(disp_arr.astype(np.float32)), disp_pt)

    warped_seg_path = None
    if warped_seg_arr is not None:
        _enter_phase(job_id, "segmentation")
        warped_seg_path = os.path.join(job_dir, "warped_seg.nii.gz")
        _save_nifti(warped_seg_arr, warped_seg_path, dtype=np.uint8)

    return warped_path, disp_pt, warped_seg_path


def _run_pipeline(job_id, fixed_path, moving_path, moving_seg_path):
    job = _get(job_id)
    registration_type = job["registration_type"]
    job_dir = os.path.dirname(fixed_path)
    log_path = os.path.join(job_dir, "niftyreg.log")
    try:
        if registration_type in ("affine", "deformable"):
            warped_path, disp_pt, warped_seg_path = _run_niftyreg_pipeline(
                job_id, job, fixed_path, moving_path, moving_seg_path, job_dir, log_path
            )
        elif registration_type == "syn":
            warped_path, disp_pt, warped_seg_path = _run_inprocess_pipeline(
                job_id, job, fixed_path, moving_path, moving_seg_path, job_dir,
                phase_key="syn",
                run_fn=lambda fixed_arr, moving_arr, seg_arr: run_syn_registration(
                    fixed_arr, moving_arr, seg_arr, max_iterations=job.get("max_iterations")
                ),
            )
        elif registration_type == "demons":
            warped_path, disp_pt, warped_seg_path = _run_inprocess_pipeline(
                job_id, job, fixed_path, moving_path, moving_seg_path, job_dir,
                phase_key="demons",
                run_fn=lambda fixed_arr, moving_arr, seg_arr: run_demons_registration(
                    fixed_arr, moving_arr, seg_arr,
                    iterations=job.get("max_iterations"),
                    smoothing=job.get("smoothing"),
                ),
            )
        else:
            raise RegistrationError(f"Unsupported registration_type: {registration_type}")

        results = {
            "warped_image": os.path.basename(warped_path),
            "displacement": os.path.basename(disp_pt),
            "warped_segmentation": os.path.basename(warped_seg_path) if warped_seg_path else None,
        }
        _set(job_id, status="done", results=results)
    except Exception as e:
        _set(job_id, status="error", error=str(e))


def start_job(job_id, job_dir, fixed_bytes, fixed_name, moving_bytes, moving_name, moving_seg_bytes=None, moving_seg_name=None):
    fixed_path = os.path.join(job_dir, "fixed" + _ext(fixed_name))
    moving_path = os.path.join(job_dir, "moving" + _ext(moving_name))
    with open(fixed_path, "wb") as f:
        f.write(fixed_bytes)
    with open(moving_path, "wb") as f:
        f.write(moving_bytes)

    moving_seg_path = None
    if moving_seg_bytes is not None:
        moving_seg_path = os.path.join(job_dir, "moving_seg" + _ext(moving_seg_name))
        with open(moving_seg_path, "wb") as f:
            f.write(moving_seg_bytes)

    _set(job_id, has_segmentation=moving_seg_path is not None)
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, fixed_path, moving_path, moving_seg_path),
        daemon=True,
    )
    thread.start()


def _ext(filename):
    if filename.endswith(".nii.gz"):
        return ".nii.gz"
    if filename.endswith(".nii"):
        return ".nii"
    return os.path.splitext(filename)[1] or ".nii.gz"
