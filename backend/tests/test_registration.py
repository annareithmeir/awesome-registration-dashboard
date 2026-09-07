import os
import sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app import registration, processing


def _shifted_square_pair(shape=(64, 64), shift=(4, 6)):
    """A bright square in `fixed`, the same square shifted by `shift` (a
    small offset, well within what Demons' purely-local optimization can
    recover without a prior affine step) in `moving`, plus a matching
    single-label segmentation for `moving`.
    """
    fixed = np.zeros(shape, dtype=np.float32)
    moving = np.zeros(shape, dtype=np.float32)
    moving_seg = np.zeros(shape, dtype=np.uint8)

    lo = [s // 2 - 10 for s in shape]
    hi = [s // 2 + 10 for s in shape]
    fixed[lo[0]:hi[0], lo[1]:hi[1]] = 1.0

    mlo = [lo[i] + shift[i] for i in range(2)]
    mhi = [hi[i] + shift[i] for i in range(2)]
    moving[mlo[0]:mhi[0], mlo[1]:mhi[1]] = 1.0
    moving_seg[mlo[0]:mhi[0], mlo[1]:mhi[1]] = 1

    return fixed, moving, moving_seg


def test_syn_registration_recovers_synthetic_shift():
    fixed, moving, _ = _shifted_square_pair()
    warped, disp, warped_seg = registration.run_syn_registration(fixed, moving, max_iterations=100)

    assert warped.shape == fixed.shape
    assert disp.shape == fixed.shape + (2,)
    assert warped_seg is None

    dice = processing.dice_score((warped > 0.5).astype(np.uint8), (fixed > 0.5).astype(np.uint8))
    assert dice > 0.9


def test_syn_registration_warps_segmentation():
    fixed, moving, moving_seg = _shifted_square_pair()
    warped, disp, warped_seg = registration.run_syn_registration(
        fixed, moving, moving_seg_arr=moving_seg, max_iterations=100
    )

    assert warped_seg is not None
    assert warped_seg.shape == fixed.shape
    assert set(np.unique(warped_seg).tolist()) <= {0, 1}
    dice = processing.dice_score(warped_seg, (fixed > 0.5).astype(np.uint8))
    assert dice > 0.9


def test_syn_registration_displacement_channel_order():
    # channel 0 = X/column displacement, channel 1 = Y/row displacement -
    # this app's convention (see registration.py's module docstring), which
    # is the *opposite* of ANTs' own field convention. moving is shifted
    # +6 columns, +4 rows relative to fixed, so the recovered field at the
    # fixed square's location should point in that same direction.
    fixed, moving, _ = _shifted_square_pair(shift=(4, 6))
    _, disp, _ = registration.run_syn_registration(fixed, moving, max_iterations=100)

    center = tuple(s // 2 for s in fixed.shape)
    dx, dy = disp[center]
    assert dx == pytest.approx(6, abs=1.5)
    assert dy == pytest.approx(4, abs=1.5)


def test_demons_registration_recovers_synthetic_shift():
    # A smaller shift than the SyN tests above use - Demons is a purely
    # local, diffusion-based method (see run_demons_registration's
    # docstring), so it doesn't have SyN's global search step and needs a
    # smaller displacement (relative to image/structure size) to fully
    # converge in a short run.
    fixed, moving, _ = _shifted_square_pair(shift=(2, 3))
    warped, disp, warped_seg = registration.run_demons_registration(fixed, moving, iterations=50, smoothing=1.0)

    assert warped.shape == fixed.shape
    assert disp.shape == fixed.shape + (2,)
    assert warped_seg is None

    dice = processing.dice_score((warped > 0.5).astype(np.uint8), (fixed > 0.5).astype(np.uint8))
    assert dice > 0.9


def test_demons_registration_warps_segmentation():
    fixed, moving, moving_seg = _shifted_square_pair(shift=(2, 3))
    warped, disp, warped_seg = registration.run_demons_registration(
        fixed, moving, moving_seg_arr=moving_seg, iterations=50, smoothing=1.0
    )

    assert warped_seg is not None
    assert warped_seg.shape == fixed.shape
    assert set(np.unique(warped_seg).tolist()) <= {0, 1}
    dice = processing.dice_score(warped_seg, (fixed > 0.5).astype(np.uint8))
    assert dice > 0.9


def test_demons_registration_displacement_channel_order():
    # Same convention check as SyN above, but for SimpleITK's field, which
    # (unlike ANTs') already matches this app's convention with no
    # reordering needed - if a future SimpleITK/refactor ever changed that,
    # this would catch it. Demons' field doesn't converge to the exact
    # shift magnitude the way SyN's optimizer-driven one does (it's a
    # diffusion process, not a direct fit), so this checks direction and
    # relative channel order - both positive, and larger in the axis that
    # was actually shifted more (columns: +6 vs rows: +4) - averaged over
    # the whole shifted region rather than a single point, since the field
    # is weakest at that region's own center (no local image gradient for a
    # uniform-intensity square to drive it, deep inside the shape).
    fixed, moving, _ = _shifted_square_pair(shape=(64, 64), shift=(4, 6))
    _, disp, _ = registration.run_demons_registration(fixed, moving, iterations=100, smoothing=1.0)

    lo = [s // 2 - 10 for s in fixed.shape]
    hi = [s // 2 + 10 for s in fixed.shape]
    dx, dy = disp[lo[0]:hi[0], lo[1]:hi[1]].mean(axis=(0, 1))
    assert dx > 0
    assert dy > 0
    assert dx > dy


def test_create_job_accepts_all_registration_types():
    for reg_type in registration.REGISTRATION_TYPES:
        job_id, job_dir = registration.create_job("/tmp", registration_type=reg_type)
        assert job_id
        os.rmdir(job_dir)


def test_create_job_rejects_unknown_registration_type():
    with pytest.raises(registration.RegistrationError):
        registration.create_job("/tmp", registration_type="not-a-real-method")
