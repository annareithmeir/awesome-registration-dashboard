import io
import os
import tempfile
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial.distance import cdist

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
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def dice95(a: np.ndarray, b: np.ndarray):
    a = _binarize(a)
    b = _binarize(b)
    if a.ndim == 2:
        return dice_score(a, b)
    dices = [dice_score(a[i], b[i]) for i in range(a.shape[0])]
    return float(np.percentile(dices, 95))


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
    da = cdist(pa, pb)
    db = cdist(pb, pa)
    d_ab = np.max(np.min(da, axis=1))
    d_ba = np.max(np.min(db, axis=1))
    return float(max(d_ab, d_ba))


def hausdorff95(a: np.ndarray, b: np.ndarray):
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
                "dice95": dice95(a_mask, b_mask),
                "hausdorff": hausdorff(a_mask, b_mask),
                "hausdorff95": hausdorff95(a_mask, b_mask),
                "fixed_voxels": int(a_mask.sum()),
                "moving_voxels": int(b_mask.sum()),
            }
        )
    return results
