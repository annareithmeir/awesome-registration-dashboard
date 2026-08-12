import numpy as np
import nibabel as nib
import os
import torch


def generate_circle_expansion_2d(shape=(64, 64), radius=16, expansion=8):
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    expand_center = (22, 22)
    shrink_center = (43, 43)

    # Use the same centers and smooth local field as the dedicated displacement sample.
    disp = generate_local_expand_shrink_disp_2d(
        shape=shape,
        expansion_center=expand_center,
        shrink_center=shrink_center,
    )

    def radial_distance(center):
        cy, cx = center
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    def soft_disk(r, object_radius, edge_width=1.8):
        return 1.0 / (1.0 + np.exp((r - object_radius) / max(edge_width, 1e-3)))

    def region_texture(r, object_radius, core_level, rim_level):
        normalized = r / max(object_radius, 1.0)
        core = core_level + 0.28 * np.exp(-(normalized ** 2) / (2.0 * 0.34 ** 2))
        rim = rim_level * np.exp(-((normalized - 0.78) ** 2) / (2.0 * 0.12 ** 2))
        return core + rim

    grow_fixed_radius = max(5.0, radius * 0.68)
    grow_moving_radius = grow_fixed_radius + 0.42 * expansion
    shrink_fixed_radius = max(6.0, radius * 1.00)
    shrink_moving_radius = max(3.0, shrink_fixed_radius - 0.42 * expansion)

    grow_r = radial_distance(expand_center)
    shrink_r = radial_distance(shrink_center)

    background = 0.10
    background += 0.04 * np.exp(-(grow_r ** 2) / (2.0 * (grow_fixed_radius + 7.0) ** 2))
    background += 0.03 * np.exp(-(shrink_r ** 2) / (2.0 * (shrink_fixed_radius + 7.0) ** 2))

    fixed_img = background.copy()
    fixed_img += soft_disk(grow_r, grow_fixed_radius) * (region_texture(grow_r, grow_fixed_radius, 0.22, 0.10) - background)
    fixed_img += soft_disk(shrink_r, shrink_fixed_radius) * (region_texture(shrink_r, shrink_fixed_radius, 0.54, 0.06) - background)

    moving_img = background.copy()
    moving_img += soft_disk(grow_r, grow_moving_radius) * (region_texture(grow_r, grow_moving_radius, 0.22, 0.10) - background)
    moving_img += soft_disk(shrink_r, shrink_moving_radius) * (region_texture(shrink_r, shrink_moving_radius, 0.54, 0.06) - background)

    fixed_img = np.clip(fixed_img, 0.0, 1.0).astype(np.float32)
    moving_img = np.clip(moving_img, 0.0, 1.0).astype(np.float32)

    seg_fixed = np.zeros((H, W), dtype=np.uint8)
    seg_moving = np.zeros((H, W), dtype=np.uint8)
    seg_fixed[grow_r <= grow_fixed_radius] = 1
    seg_moving[grow_r <= grow_moving_radius] = 1
    seg_fixed[shrink_r <= shrink_fixed_radius] = 2
    seg_moving[shrink_r <= shrink_moving_radius] = 2

    return fixed_img, moving_img, seg_fixed, seg_moving, disp


def generate_bw_image(shape=(64, 64)):
    H, W = shape
    img = np.zeros((H, W), dtype=np.float32)
    img[:, : W // 2] = 0.0
    img[:, W // 2 :] = 1.0
    seg = np.zeros((H, W), dtype=np.uint8)
    seg[:, W // 2 :] = 1
    return img, seg


def generate_3d_image(shape=(32, 64, 64), radius=15):
    D, H, W = shape
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    cz, cy, cx = D / 2, H / 2, W / 2
    img = np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * (radius ** 2)))
    seg = img > img.mean()
    return img.astype(np.float32), seg.astype(np.uint8)


def generate_radial_disp_2d(shape=(64, 64), scale=2.0):
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    vy = (y - cy) / (np.sqrt((y - cy) ** 2 + (x - cx) ** 2) + 1e-6)
    vx = (x - cx) / (np.sqrt((y - cy) ** 2 + (x - cx) ** 2) + 1e-6)
    disp = np.stack([vx, vy], axis=-1) * scale
    return disp.astype(np.float32)


def generate_random_disp_2d(shape=(64, 64), scale=3.0, seed=123):
    H, W = shape
    rng = np.random.RandomState(seed)
    disp = rng.uniform(-1.0, 1.0, size=(H, W, 2)).astype(np.float32) * scale
    return disp


def generate_local_expand_shrink_disp_2d(
    shape=(64, 64), expansion_center=(22, 22), shrink_center=(43, 43), sigma=8.0
):
    """Create a smooth 2D field with one locally expanding and one contracting region.

    The output is channel-last ``(H, W, 2)`` in ``(x, y)`` component order.
    """
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]

    def local_radial_field(center, strength):
        cy, cx = center
        dx = x - cx
        dy = y - cy
        weight = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
        return strength * dx * weight, strength * dy * weight

    # Positive radial strength expands locally; negative strength contracts locally.
    expand_x, expand_y = local_radial_field(expansion_center, strength=0.32)
    shrink_x, shrink_y = local_radial_field(shrink_center, strength=-0.28)
    return np.stack([expand_x + shrink_x, expand_y + shrink_y], axis=-1).astype(np.float32)


def generate_radial_disp_3d(shape=(32, 64, 64), scale=1.0):
    D, H, W = shape
    z, y, x = np.mgrid[0:D, 0:H, 0:W]
    cz, cy, cx = D / 2, H / 2, W / 2
    rz = z - cz
    ry = y - cy
    rx = x - cx
    r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2) + 1e-6
    uz = rz / r
    uy = ry / r
    ux = rx / r
    disp = np.stack([ux, uy, uz], axis=-1) * scale
    return disp.astype(np.float32)


def save_nifti(arr, path):
    img = nib.Nifti1Image(arr, np.eye(4))
    nib.save(img, path)


def save_pt(arr, path, channel_first=False):
    if channel_first:
        arr_s = np.moveaxis(arr, -1, 0)
    else:
        arr_s = arr
    torch.save(torch.from_numpy(arr_s), path)


def generate_and_save_samples(out_dir="backend/test_samples"):
    os.makedirs(out_dir, exist_ok=True)
    img2d_fixed, img2d_moving, seg2d_fixed, seg2d_moving, disp2d = generate_circle_expansion_2d()
    save_nifti(img2d_fixed, os.path.join(out_dir, "img2d_fixed.nii.gz"))
    save_nifti(img2d_moving, os.path.join(out_dir, "img2d_moving.nii.gz"))
    save_nifti(seg2d_fixed.astype(np.uint8), os.path.join(out_dir, "seg2d_fixed.nii.gz"))
    save_nifti(seg2d_moving.astype(np.uint8), os.path.join(out_dir, "seg2d_moving.nii.gz"))
    save_pt(disp2d, os.path.join(out_dir, "disp2d_chlast.pt"), channel_first=False)
    save_pt(disp2d, os.path.join(out_dir, "disp2d_chfirst.pt"), channel_first=True)

    img2d_bw, seg2d_bw = generate_bw_image()
    disp2d_rand = generate_random_disp_2d()
    disp2d_expand_shrink = generate_local_expand_shrink_disp_2d()
    save_nifti(img2d_bw, os.path.join(out_dir, "img2d_bw.nii.gz"))
    save_nifti(seg2d_bw.astype(np.uint8), os.path.join(out_dir, "seg2d_bw.nii.gz"))
    save_pt(disp2d_rand, os.path.join(out_dir, "disp2d_random.pt"), channel_first=False)
    save_pt(
        disp2d_expand_shrink,
        os.path.join(out_dir, "disp2d_local_expand_shrink.pt"),
        channel_first=False,
    )

    img3d, seg3d = generate_3d_image()
    disp3d = generate_radial_disp_3d()
    save_nifti(img3d, os.path.join(out_dir, "img3d.nii.gz"))
    save_nifti(seg3d.astype(np.uint8), os.path.join(out_dir, "seg3d.nii.gz"))
    save_pt(disp3d, os.path.join(out_dir, "disp3d_chlast.pt"), channel_first=False)
    save_pt(disp3d, os.path.join(out_dir, "disp3d_chfirst.pt"), channel_first=True)
