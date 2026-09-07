import io
import os
import tempfile
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial.distance import cdist, dice as _scipy_dice_distance, directed_hausdorff

try:
    import torch
except Exception:
    torch = None


def _squeeze_to_array(arr):
    arr = np.asarray(arr)
    return np.squeeze(arr)


def load_nifti_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        arr = img.get_fdata(dtype=np.float32)
        return arr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_py_tensor(data: bytes):
    if torch is None:
        raise RuntimeError("PyTorch is required to load .pt files")
    buf = io.BytesIO(data)
    tensor = torch.load(buf, map_location="cpu")
    if hasattr(tensor, "numpy"):
        arr = tensor.numpy()
    else:
        arr = np.array(tensor)
    return arr


def load_pt_bytes(data: bytes):
    arr = _load_py_tensor(data)
    arr = _squeeze_to_array(arr)
    if arr.ndim < 2:
        raise ValueError("Displacement tensor has too few dimensions")
    if arr.ndim not in (2, 3, 4):
        raise ValueError(f"Unsupported tensor dimensions {arr.ndim}")
    return arr.astype(np.float32)


def normalize_disp(arr: np.ndarray):
    arr = _squeeze_to_array(arr)
    if arr.ndim == 2:
        raise ValueError("Displacement tensor needs a channel dimension")
    if arr.shape[0] in (2, 3):
        arr = np.moveaxis(arr, 0, -1)
    elif arr.shape[-1] in (2, 3):
        pass
    else:
        raise ValueError("Unable to infer displacement channel dimension; expected 2 or 3 channels")
    if arr.ndim not in (3, 4):
        raise ValueError(f"Unexpected displacement shape after normalization: {arr.shape}")
    return arr.astype(np.float32)


def compute_jacobian_det(disp: np.ndarray, spacing=None):
    disp = np.asarray(disp, dtype=np.float32)
    if spacing is None:
        spacing = [1.0] * (disp.ndim - 1)
    if disp.ndim == 3:
        H, W, C = disp.shape
        if C != 2:
            raise ValueError("2D displacement must have 2 channels")
        ux = disp[..., 0]
        uy = disp[..., 1]
        duy_dy, duy_dx = np.gradient(uy, spacing[0], spacing[1])
        dux_dy, dux_dx = np.gradient(ux, spacing[0], spacing[1])
        Jxx = 1.0 + dux_dx
        Jxy = dux_dy
        Jyx = duy_dx
        Jyy = 1.0 + duy_dy
        det = Jxx * Jyy - Jxy * Jyx
        return det
    elif disp.ndim == 4:
        D, H, W, C = disp.shape
        if C != 3:
            raise ValueError("3D displacement must have 3 channels")
        ux = disp[..., 0]
        uy = disp[..., 1]
        uz = disp[..., 2]
        dux_dz, dux_dy, dux_dx = np.gradient(ux, spacing[0], spacing[1], spacing[2])
        duy_dz, duy_dy, duy_dx = np.gradient(uy, spacing[0], spacing[1], spacing[2])
        duz_dz, duz_dy, duz_dx = np.gradient(uz, spacing[0], spacing[1], spacing[2])
        a = 1.0 + dux_dx
        b = dux_dy
        c = dux_dz
        d = duy_dx
        e = 1.0 + duy_dy
        f = duy_dz
        g = duz_dx
        h = duz_dy
        i = 1.0 + duz_dz
        det = (
            a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g)
        )
        return det
    else:
        raise ValueError(f"Unsupported displacement array ndim={disp.ndim}")


def _local_jacobian_matrix(disp: np.ndarray, spacing=None):
    """Per-voxel local Jacobian matrix J = I + grad(u), shape (..., C, C)."""
    disp = np.asarray(disp, dtype=np.float32)
    if spacing is None:
        spacing = [1.0] * (disp.ndim - 1)
    if disp.ndim == 3:
        H, W, C = disp.shape
        if C != 2:
            raise ValueError("2D displacement must have 2 channels")
        ux = disp[..., 0]
        uy = disp[..., 1]
        duy_dy, duy_dx = np.gradient(uy, spacing[0], spacing[1])
        dux_dy, dux_dx = np.gradient(ux, spacing[0], spacing[1])
        J = np.empty((H, W, 2, 2), dtype=np.float32)
        J[..., 0, 0] = 1.0 + dux_dx
        J[..., 0, 1] = dux_dy
        J[..., 1, 0] = duy_dx
        J[..., 1, 1] = 1.0 + duy_dy
        return J
    elif disp.ndim == 4:
        D, H, W, C = disp.shape
        if C != 3:
            raise ValueError("3D displacement must have 3 channels")
        ux = disp[..., 0]
        uy = disp[..., 1]
        uz = disp[..., 2]
        dux_dz, dux_dy, dux_dx = np.gradient(ux, spacing[0], spacing[1], spacing[2])
        duy_dz, duy_dy, duy_dx = np.gradient(uy, spacing[0], spacing[1], spacing[2])
        duz_dz, duz_dy, duz_dx = np.gradient(uz, spacing[0], spacing[1], spacing[2])
        J = np.empty((D, H, W, 3, 3), dtype=np.float32)
        J[..., 0, 0] = 1.0 + dux_dx
        J[..., 0, 1] = dux_dy
        J[..., 0, 2] = dux_dz
        J[..., 1, 0] = duy_dx
        J[..., 1, 1] = 1.0 + duy_dy
        J[..., 1, 2] = duy_dz
        J[..., 2, 0] = duz_dx
        J[..., 2, 1] = duz_dy
        J[..., 2, 2] = 1.0 + duz_dz
        return J
    else:
        raise ValueError(f"Unsupported displacement array ndim={disp.ndim}")


def compute_shear_index(disp: np.ndarray, spacing=None):
    """Local anisotropy of the deformation: ratio of largest to smallest
    singular value of the local Jacobian matrix. 1.0 means locally
    rigid/isotropic; larger values mean shearing or stretching that a
    volume-only measure like the Jacobian determinant can miss (e.g. a
    matrix can have det==1 while still shearing heavily).
    """
    J = _local_jacobian_matrix(disp, spacing)
    singular_values = np.linalg.svd(J, compute_uv=False)  # shape (..., C), descending
    s_max = singular_values[..., 0]
    s_min = singular_values[..., -1]
    shear = np.divide(
        s_max,
        s_min,
        out=np.full_like(s_max, np.inf),
        where=s_min > 1e-8,
    )
    return shear.astype(np.float32)


def compute_inverse_displacement(disp: np.ndarray, iterations: int = 6):
    """Numerically approximate the inverse of a displacement field via
    fixed-point iteration: v_(k+1)(x) = -u(x + v_k(x)). Works for any
    smooth, mildly-deformed field (typical regularized registration output)
    without needing the original parametric transform.
    """
    disp = np.asarray(disp, dtype=np.float32)
    shape = disp.shape[:-1]
    grid = np.mgrid[tuple(slice(0, s) for s in shape)].astype(np.float32)

    inv = -disp.copy()
    for _ in range(iterations):
        sample_coords = grid + np.moveaxis(inv, -1, 0)
        sampled = np.stack(
            [
                ndimage.map_coordinates(disp[..., c], sample_coords, order=1, mode="nearest")
                for c in range(disp.shape[-1])
            ],
            axis=-1,
        )
        inv = -sampled
    return inv


def compute_inverse_consistency_error(disp: np.ndarray, iterations: int = 6):
    """Compose the forward field with its numerically-estimated inverse and
    measure the residual displacement from identity at each voxel: a
    well-behaved (diffeomorphic) transform should compose back to ~0
    everywhere; large residuals flag regions where the field is poorly
    invertible even if the Jacobian determinant alone looks fine.
    """
    disp = np.asarray(disp, dtype=np.float32)
    inv = compute_inverse_displacement(disp, iterations=iterations)
    shape = disp.shape[:-1]
    grid = np.mgrid[tuple(slice(0, s) for s in shape)].astype(np.float32)
    warped_coords = grid + np.moveaxis(disp, -1, 0)
    inv_at_warped = np.stack(
        [
            ndimage.map_coordinates(inv[..., c], warped_coords, order=1, mode="nearest")
            for c in range(disp.shape[-1])
        ],
        axis=-1,
    )
    ice_vector = disp + inv_at_warped
    return np.linalg.norm(ice_vector, axis=-1).astype(np.float32)


def sign_map(det: np.ndarray):
    sm = np.zeros_like(det, dtype=np.int8)
    sm[det > 0] = 1
    sm[det < 0] = -1
    return sm


def _binarize(mask: np.ndarray):
    return (mask > 0).astype(np.bool_)


def dice_score(a: np.ndarray, b: np.ndarray):
    a = _binarize(a)
    b = _binarize(b)
    if not a.any() and not b.any():
        # scipy's dice distance is 0/0 (nan, with a warning) here - both
        # empty is a trivial perfect match by convention.
        return 1.0
    # scipy.spatial.distance.dice returns the *dissimilarity*
    # (ntf+nft)/(2*ntt+ntf+nft); the Dice similarity coefficient is 1 minus
    # that.
    return float(1.0 - _scipy_dice_distance(a.ravel(), b.ravel()))


def dice30(a: np.ndarray, b: np.ndarray):
    """30th percentile of per-slice Dice scores through a volume - a
    stricter summary than the plain (whole-volume) Dice above it, since it
    reports roughly "the worse third of slices are at or below this",
    rather than a single number averaged over every slice regardless of how
    badly some of them match. For a single 2D slice there's nothing to take
    a percentile over, so this is just the plain Dice score.
    """
    a = _binarize(a)
    b = _binarize(b)
    if a.ndim == 2:
        return dice_score(a, b)
    dices = [dice_score(a[i], b[i]) for i in range(a.shape[0])]
    return float(np.percentile(dices, 30))


def _surface_points(mask: np.ndarray):
    mask = _binarize(mask)
    eroded = ndimage.binary_erosion(mask)
    surf = mask ^ eroded
    coords = np.array(np.nonzero(surf)).T
    if coords.size == 0:
        return np.empty((0, mask.ndim))
    return coords


def hausdorff(a: np.ndarray, b: np.ndarray):
    pa = _surface_points(a)
    pb = _surface_points(b)
    if pa.size == 0 and pb.size == 0:
        return 0.0
    if pa.size == 0 or pb.size == 0:
        return float("inf")
    # The (symmetric) Hausdorff distance is the max of the two directed
    # Hausdorff distances; scipy.spatial.distance.directed_hausdorff computes
    # each one (via a linear-random-search algorithm, not a full pairwise
    # distance matrix - faster than the cdist approach below for large point
    # sets).
    d_ab = directed_hausdorff(pa, pb)[0]
    d_ba = directed_hausdorff(pb, pa)[0]
    return float(max(d_ab, d_ba))


def hausdorff95(a: np.ndarray, b: np.ndarray):
    # Unlike plain Hausdorff above, this needs the full distribution of
    # nearest-neighbor distances (to take its 95th percentile), not just its
    # max - directed_hausdorff only ever returns the max, so there's no
    # library shortcut here; this stays a full pairwise distance matrix via
    # cdist.
    pa = _surface_points(a)
    pb = _surface_points(b)
    if pa.size == 0 and pb.size == 0:
        return 0.0
    if pa.size == 0 or pb.size == 0:
        return float("inf")
    da = cdist(pa, pb)
    db = cdist(pb, pa)
    d_ab = np.percentile(np.min(da, axis=1), 95)
    d_ba = np.percentile(np.min(db, axis=1), 95)
    return float(max(d_ab, d_ba))


def frac_negative_jacobian(det: np.ndarray):
    total = det.size
    if total == 0:
        return 0.0
    neg = (det < 0).sum()
    return float(neg) / float(total)


def log_jacobian_stats(det: np.ndarray):
    """Mean and std of log(J) over voxels with J > 0. Unlike raw J — bounded
    below by 0 but unbounded above, so growth voxels dominate a plain
    mean/std — log(J) is symmetric around 0 for equal growth vs. shrinkage,
    which is why it's the standard tensor-based-morphometry summary of local
    volume change. Voxels with J <= 0 (folding) are undefined under log and
    excluded; excluded_fraction reports how much of the field that was.
    """
    det = np.asarray(det, dtype=np.float32)
    total = det.size
    if total == 0:
        return {"mean": 0.0, "std": 0.0, "excluded_fraction": 0.0}
    positive = det[det > 0]
    if positive.size == 0:
        return {"mean": 0.0, "std": 0.0, "excluded_fraction": 1.0}
    log_det = np.log(positive)
    return {
        "mean": float(np.mean(log_det)),
        "std": float(np.std(log_det)),
        "excluded_fraction": float(1.0 - (positive.size / total)),
    }


def per_label_metrics(a: np.ndarray, b: np.ndarray, labels=None):
    a = np.asarray(a)
    b = np.asarray(b)
    if labels is None:
        labels = sorted(set(np.unique(a).tolist()) | set(np.unique(b).tolist()))
        labels = [int(label) for label in labels if int(label) > 0]

    results = []
    for label in labels:
        a_mask = (a == label).astype(np.uint8)
        b_mask = (b == label).astype(np.uint8)
        results.append(
            {
                "label": int(label),
                "dice": dice_score(a_mask, b_mask),
                "dice30": dice30(a_mask, b_mask),
                "hausdorff": hausdorff(a_mask, b_mask),
                "hausdorff95": hausdorff95(a_mask, b_mask),
                "fixed_voxels": int(a_mask.sum()),
                "moving_voxels": int(b_mask.sum()),
            }
        )
    return results


def aggregate_label_metrics(per_label):
    """The "overall" Dice/Dice30/Hausdorff/Hausdorff95 shown alongside a
    per-structure breakdown, computed as the mean of each metric across the
    reported structures - so it's actually consistent with (the mean of)
    that breakdown.

    This is deliberately NOT the same as computing dice_score/hausdorff
    directly on the labels merged into one binary mask: merging first and
    comparing second measures overlap of the union of all structures, which
    is a different quantity from the average of each structure's own
    overlap, and the two generally don't agree (e.g. two structures that
    each individually overlap 50% can merge into a union that overlaps
    anywhere from 0% to 100%, depending on whether their mismatched regions
    coincide). Returns None when there are no labeled structures to
    average, so the caller can fall back to comparing the images directly.
    """
    if not per_label:
        return None
    return {
        "dice": float(np.mean([item["dice"] for item in per_label])),
        "dice30": float(np.mean([item["dice30"] for item in per_label])),
        "hausdorff": float(np.mean([item["hausdorff"] for item in per_label])),
        "hausdorff95": float(np.mean([item["hausdorff95"] for item in per_label])),
    }
