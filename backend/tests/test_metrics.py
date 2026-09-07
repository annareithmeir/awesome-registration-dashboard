import os
import sys
import numpy as np
import pytest
from pathlib import Path
from scipy.spatial.distance import cdist
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app import processing, test_data


def test_dice_perfect():
    a = np.zeros((32, 32), dtype=np.uint8)
    a[8:24, 8:24] = 1
    b = a.copy()
    assert processing.dice_score(a, b) == 1.0


def test_dice30_3d():
    _, seg = test_data.generate_3d_image((8, 16, 16))
    assert processing.dice30(seg, seg) == 1.0


def test_dice30_slice():
    a = np.zeros((4, 8, 8), dtype=np.uint8)
    b = np.zeros((4, 8, 8), dtype=np.uint8)
    a[0, 2:6, 2:6] = 1
    b[0, 2:6, 2:6] = 1
    a[3, 1:5, 1:5] = 1
    b[3, 1:5, 1:5] = 1
    assert processing.dice30(a, b) == 1.0


def test_dice30_percentile_of_varying_slices():
    # Per-slice Dice scores of 0.2, 0.4, 0.6, 0.8 by construction (each
    # slice: 10-voxel fixed region, moving region overlapping it by
    # 2/4/6/8 voxels respectively, both size 10 -> dice = 2*inter/20). The
    # 30th percentile of [0.2, 0.4, 0.6, 0.8] is 0.38 (numpy's default
    # linear interpolation, between the 0.2 and 0.4 slices) - not simply
    # the lowest, or one of the four values, so this actually exercises the
    # percentile rather than a degenerate all-identical case.
    a = np.zeros((4, 1, 20), dtype=np.uint8)
    b = np.zeros((4, 1, 20), dtype=np.uint8)
    a[:, 0, 0:10] = 1
    for i, inter in enumerate([2, 4, 6, 8]):
        b[i, 0, 0:inter] = 1
        b[i, 0, 10:10 + (10 - inter)] = 1
    assert processing.dice30(a, b) == pytest.approx(0.38)


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


def test_dice_score_hand_computed_partial_overlap():
    # Two 2x2 blocks, shifted by one column: intersection is the 2x1 strip
    # they share (2 voxels), each block is 4 voxels -> 2*2/(4+4) = 0.5.
    a = np.zeros((4, 4), dtype=np.uint8)
    a[0:2, 0:2] = 1
    b = np.zeros((4, 4), dtype=np.uint8)
    b[0:2, 1:3] = 1
    assert processing.dice_score(a, b) == pytest.approx(0.5)


def test_dice_score_matches_naive_formula():
    # Independent re-derivation of the textbook Dice formula (not calling
    # scipy, not sharing any code with processing.dice_score) as a
    # cross-check on random inputs, not just hand-picked ones.
    rng = np.random.RandomState(0)
    for _ in range(20):
        a = rng.rand(20, 20) > 0.5
        b = rng.rand(20, 20) > 0.5
        inter = np.logical_and(a, b).sum()
        denom = a.sum() + b.sum()
        expected = 1.0 if denom == 0 else 2.0 * inter / denom
        assert processing.dice_score(a.astype(np.uint8), b.astype(np.uint8)) == pytest.approx(expected)


def test_dice_score_matches_scipy_dice_distance():
    # Documents (and pins) the library function dice_score is built on.
    from scipy.spatial.distance import dice as scipy_dice_distance
    rng = np.random.RandomState(1)
    a = (rng.rand(15, 15) > 0.4).astype(np.uint8)
    b = (rng.rand(15, 15) > 0.6).astype(np.uint8)
    expected = 1.0 - scipy_dice_distance(a.astype(bool).ravel(), b.astype(bool).ravel())
    assert processing.dice_score(a, b) == pytest.approx(expected)


def test_hausdorff_matches_bruteforce_cdist_reference():
    # Independent brute-force reference (full pairwise distance matrix, not
    # scipy's directed_hausdorff) for the same random blobs, to confirm
    # switching hausdorff() over to directed_hausdorff didn't change its
    # results.
    rng = np.random.RandomState(2)
    for _ in range(8):
        a = np.zeros((24, 24), dtype=np.uint8)
        b = np.zeros((24, 24), dtype=np.uint8)
        ax, ay = rng.randint(2, 14, size=2)
        bx, by = rng.randint(2, 14, size=2)
        a[ax:ax + 8, ay:ay + 8] = 1
        b[bx:bx + 8, by:by + 8] = 1

        pa = processing._surface_points(a)
        pb = processing._surface_points(b)
        da = cdist(pa, pb)
        db = cdist(pb, pa)
        expected = max(np.max(np.min(da, axis=1)), np.max(np.min(db, axis=1)))
        assert processing.hausdorff(a, b) == pytest.approx(expected)


def test_aggregate_metrics_are_mean_of_per_label_not_merged_mask():
    # This is the reported bug: the "overall" Dice/HD used to be computed by
    # merging all labels into one binary mask and comparing that, which is a
    # different (generally different-valued) quantity than the mean of the
    # per-structure scores. Label 1 matches perfectly (dice=1.0); label 2 is
    # shifted completely out of overlap (dice=0.0) - mean is 0.5, but merging
    # both labels into one mask before comparing gives 0.8 (see below),
    # since label 1 dominates the merged mask by voxel count.
    a = np.zeros((10, 10), dtype=np.uint8)
    a[0:4, 0:4] = 1  # label 1: 16 voxels
    a[5:7, 5:7] = 2  # label 2: 4 voxels

    b = np.zeros((10, 10), dtype=np.uint8)
    b[0:4, 0:4] = 1  # label 1: exact match
    b[5:7, 8:10] = 2  # label 2: shifted completely out of overlap with a's label 2

    per_label = processing.per_label_metrics(a, b)
    assert per_label[0]["dice"] == pytest.approx(1.0)
    assert per_label[1]["dice"] == pytest.approx(0.0)

    aggregate = processing.aggregate_label_metrics(per_label)
    assert aggregate["dice"] == pytest.approx(0.5)

    # The old (buggy) behavior, kept here only to prove it's a genuinely
    # different number - not something the fix coincidentally also produces.
    merged_mask_dice = processing.dice_score(a, b)
    assert merged_mask_dice == pytest.approx(0.8)
    assert aggregate["dice"] != pytest.approx(merged_mask_dice)


def test_aggregate_label_metrics_empty_returns_none():
    assert processing.aggregate_label_metrics([]) is None


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
    sample_path = Path(__file__).resolve().parents[1] / "test_samples" / "img2d_fixed.nii.gz"
    data = sample_path.read_bytes()
    arr = processing.load_nifti_bytes(data)
    assert arr.ndim == 2
    assert arr.shape == (64, 64)
