import os
import sys
import numpy as np
from pathlib import Path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app import processing, test_data


def test_dice_perfect():
    a = np.zeros((32, 32), dtype=np.uint8)
    a[8:24, 8:24] = 1
    b = a.copy()
    assert processing.dice_score(a, b) == 1.0


def test_dice95_3d():
    _, seg = test_data.generate_3d_image((8, 16, 16))
    assert processing.dice95(seg, seg) == 1.0


def test_dice95_slice():
    a = np.zeros((4, 8, 8), dtype=np.uint8)
    b = np.zeros((4, 8, 8), dtype=np.uint8)
    a[0, 2:6, 2:6] = 1
    b[0, 2:6, 2:6] = 1
    a[3, 1:5, 1:5] = 1
    b[3, 1:5, 1:5] = 1
    assert processing.dice95(a, b) == 1.0


def test_hausdorff_identical():
    a = np.zeros((16, 16), dtype=np.uint8)
    a[4:12, 4:12] = 1
    assert processing.hausdorff(a, a) == 0.0
    assert processing.hausdorff95(a, a) == 0.0


def test_jacobian_2d():
    disp = test_data.generate_radial_disp_2d((32, 32), scale=0.5)
    norm = processing.normalize_disp(disp)
    det = processing.compute_jacobian_det(norm)
    assert det.shape == (32, 32)
    assert processing.frac_negative_jacobian(det) < 0.5


def test_circle_expansion_sample_is_realistic():
    fixed_img, moving_img, seg_fixed, seg_moving, disp = test_data.generate_circle_expansion_2d((64, 64), radius=16, expansion=8)
    assert fixed_img.shape == (64, 64)
    assert moving_img.shape == (64, 64)
    assert seg_fixed.shape == (64, 64)
    assert seg_moving.shape == (64, 64)
    assert disp.shape == (64, 64, 2)
    assert 0 in np.unique(seg_fixed)
    assert 1 in np.unique(seg_fixed)
    assert 2 in np.unique(seg_fixed)
    assert 0 in np.unique(seg_moving)
    assert 1 in np.unique(seg_moving)
    assert 2 in np.unique(seg_moving)
    assert np.count_nonzero(seg_moving == 1) > np.count_nonzero(seg_fixed == 1)
    assert np.count_nonzero(seg_moving == 2) < np.count_nonzero(seg_fixed == 2)
    assert fixed_img[seg_fixed > 0].ptp() > 0.2
    assert moving_img[seg_moving > 0].ptp() > 0.2
    assert np.isfinite(disp).all()


def test_per_label_metrics_reports_foreground_structures():
    _, _, seg_fixed, seg_moving, _ = test_data.generate_circle_expansion_2d((64, 64), radius=16, expansion=8)
    per_label = processing.per_label_metrics(seg_fixed, seg_moving)

    assert [item["label"] for item in per_label] == [1, 2]
    assert per_label[0]["moving_voxels"] > per_label[0]["fixed_voxels"]
    assert per_label[1]["moving_voxels"] < per_label[1]["fixed_voxels"]
    for item in per_label:
        assert 0.0 <= item["dice"] <= 1.0
        assert item["hausdorff"] >= 0.0


def test_load_pt_channel_orders():
    disp = test_data.generate_radial_disp_2d((16, 16), scale=0.2)
    disp_cf = np.moveaxis(disp, -1, 0)
    loaded = processing.normalize_disp(disp_cf)
    assert loaded.shape == disp.shape
    np.testing.assert_allclose(loaded, disp, rtol=1e-5, atol=1e-6)

    disp_cl = disp
    loaded = processing.normalize_disp(disp_cl)
    assert loaded.shape == disp.shape
    np.testing.assert_allclose(loaded, disp, rtol=1e-5, atol=1e-6)


def test_load_nifti_from_bytes():
    sample_path = Path(__file__).resolve().parents[1] / "test_samples" / "img2d.nii.gz"
    data = sample_path.read_bytes()
    arr = processing.load_nifti_bytes(data)
    assert arr.ndim == 2
    assert arr.shape == (64, 64)
