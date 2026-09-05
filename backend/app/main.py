import os
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from . import processing, registration, test_data

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sample_dir = os.path.join(os.path.dirname(__file__), "..", "test_samples")
os.makedirs(sample_dir, exist_ok=True)
app.mount("/test_samples", StaticFiles(directory=sample_dir), name="test_samples")

registration_jobs_dir = os.path.join(os.path.dirname(__file__), "..", "registration_jobs")
os.makedirs(registration_jobs_dir, exist_ok=True)
app.mount("/registration_jobs", StaticFiles(directory=registration_jobs_dir), name="registration_jobs")


def _load_image_bytes(data: bytes, filename: str):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        return processing.load_nifti_bytes(data)
    if filename.endswith(".pt"):
        return processing.load_pt_bytes(data)
    raise ValueError("Unsupported file type: %s" % filename)


@app.post("/preview-slice")
async def preview_slice(
    file: UploadFile = File(...),
    slice_index: int = Form(0),
    mode: str = Form("image"),
    axis: int = Form(0),
):
    try:
        data = await file.read()
        arr = _load_image_bytes(data, file.filename)
        arr = np.asarray(arr, dtype=np.float32)
        # Which of the volume's (up to) three spatial axes to slice along -
        # 0/1/2 select the anatomical plane, as chosen in the frontend's
        # axis picker. Only meaningful once a volume is actually 3D.
        spatial_axis = max(0, min(axis, 2))

        if file.filename.endswith(".pt"):
            arr = processing.normalize_disp(arr)
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                preview = np.zeros(arr.shape, dtype=np.float32)
                volume_shape = arr.shape
                slice_count = 1
                is_3d = False
                yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
                warp_x = xx.astype(np.float32) + arr[..., 0]
                warp_y = yy.astype(np.float32) + arr[..., 1]
            elif arr.ndim == 3 and arr.shape[-1] in (2, 3):
                preview = np.zeros(arr.shape[:2], dtype=np.float32)
                volume_shape = arr.shape
                slice_count = 1
                is_3d = False
                yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
                warp_x = xx.astype(np.float32) + arr[..., 0]
                warp_y = yy.astype(np.float32) + arr[..., 1]
            elif arr.ndim == 4 and arr.shape[-1] in (2, 3):
                if slice_index < 0 or slice_index >= arr.shape[spatial_axis]:
                    raise HTTPException(status_code=400, detail="Slice index out of range")
                disp = np.take(arr, slice_index, axis=spatial_axis)
                # The two spatial axes left after removing `spatial_axis`
                # become the preview's (row, col) plane, in their original
                # order. Displacement channels are ordered (x, y, z), which
                # is the reverse of the volume's (axis0, axis1, axis2) =
                # (z, y, x) convention used elsewhere (see
                # processing.compute_jacobian_det) - so the channel for a
                # given remaining axis is `2 - axis`.
                row_axis, col_axis = (a for a in (0, 1, 2) if a != spatial_axis)
                row_channel, col_channel = 2 - row_axis, 2 - col_axis
                yy, xx = np.mgrid[0:disp.shape[0], 0:disp.shape[1]]
                warp_x = xx.astype(np.float32) + disp[..., col_channel]
                warp_y = yy.astype(np.float32) + disp[..., row_channel]
                preview = np.zeros((disp.shape[0], disp.shape[1]), dtype=np.float32)
                volume_shape = arr.shape
                slice_count = arr.shape[spatial_axis]
                is_3d = True
            else:
                raise HTTPException(status_code=400, detail="Unsupported displacement field dimensions")
        else:
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                preview = arr
                volume_shape = preview.shape
                slice_count = 1
                is_3d = False
            elif arr.ndim == 3:
                if slice_index < 0 or slice_index >= arr.shape[spatial_axis]:
                    raise HTTPException(status_code=400, detail="Slice index out of range")
                preview = np.take(arr, slice_index, axis=spatial_axis)
                volume_shape = arr.shape
                slice_count = arr.shape[spatial_axis]
                is_3d = True
            elif arr.ndim == 4:
                # Rare layout (e.g. an explicit channel dim) that doesn't fit
                # the plain (D, H, W) convention above - always slice along
                # the first axis regardless of the requested axis.
                if slice_index < 0 or slice_index >= arr.shape[0]:
                    raise HTTPException(status_code=400, detail="Slice index out of range")
                preview = arr[slice_index, 0]
                volume_shape = arr.shape
                slice_count = arr.shape[0]
                is_3d = True
            else:
                raise HTTPException(status_code=400, detail="Unsupported image dimensions")

        preview = np.nan_to_num(preview, nan=0.0, posinf=0.0, neginf=0.0)
        if preview.size == 0:
            preview = np.zeros((1, 1), dtype=np.float32)

        if mode == "segmentation":
            normalized = np.rint(preview).astype(np.uint8)
        else:
            min_val = float(np.min(preview))
            max_val = float(np.max(preview))
            if np.isclose(max_val, min_val):
                normalized = np.zeros_like(preview, dtype=np.uint8)
            else:
                normalized = ((preview - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

        response = {
            "shape": list(preview.shape),
            "volume_shape": list(volume_shape),
            "slice_count": slice_count,
            "is_3d": is_3d,
            "data": normalized.tolist(),
        }
        if file.filename.endswith(".pt") and warp_x is not None and warp_y is not None:
            response["warp_x"] = warp_x.tolist()
            response["warp_y"] = warp_y.tolist()

        return JSONResponse(response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _cap_non_finite(field: np.ndarray, mode: str = "zero"):
    """Replace non-finite entries so the array is safe to summarize/serialize.
    "zero" is for fields that are essentially never non-finite in practice
    (determinant, inverse-consistency error) — treat any stray value as 0.
    "cap" is for fields that can genuinely blow up at degenerate points
    (shear index, where a near-singular Jacobian gives a divide-by-~0) —
    replacing those with 0 would misleadingly read as "no distortion", so
    instead clamp them to the largest finite value actually observed.
    """
    field = np.asarray(field, dtype=np.float32)
    finite_mask = np.isfinite(field)
    if finite_mask.all():
        return field
    if mode == "cap":
        cap = float(np.max(field[finite_mask])) if finite_mask.any() else 1000.0
        return np.where(finite_mask, field, cap).astype(np.float32)
    return np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)


def _scalar_field_payload(field: np.ndarray, slice_index: int, axis: int = 0):
    if field.ndim == 2:
        preview = field
        volume_shape = field.shape
        slice_count = 1
        is_3d = False
    elif field.ndim == 3:
        spatial_axis = max(0, min(axis, 2))
        if slice_index < 0 or slice_index >= field.shape[spatial_axis]:
            raise HTTPException(status_code=400, detail="Slice index out of range")
        preview = np.take(field, slice_index, axis=spatial_axis)
        volume_shape = field.shape
        slice_count = field.shape[spatial_axis]
        is_3d = True
    else:
        raise HTTPException(status_code=400, detail="Unsupported scalar field dimensions")

    min_val = float(np.min(preview))
    max_val = float(np.max(preview))
    if np.isclose(max_val, min_val):
        normalized = np.zeros_like(preview, dtype=np.uint8)
    else:
        normalized = ((preview - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

    return {
        "min": float(np.min(field)),
        "max": float(np.max(field)),
        "mean": float(np.mean(field)),
        "shape": list(field.shape),
        "preview_shape": list(preview.shape),
        "volume_shape": list(volume_shape),
        "slice_count": slice_count,
        "is_3d": is_3d,
        "data": normalized.tolist(),
        "values": preview.tolist(),
    }


@app.post("/compute-jacobian")
async def compute_jacobian(disp_field: UploadFile = File(...), slice_index: int = Form(0), axis: int = Form(0)):
    try:
        data = await disp_field.read()
        arr = _load_image_bytes(data, disp_field.filename)
        norm = processing.normalize_disp(arr)

        det = _cap_non_finite(processing.compute_jacobian_det(norm), mode="zero")
        frac_neg = processing.frac_negative_jacobian(det)
        log_jac = processing.log_jacobian_stats(det)
        shear = _cap_non_finite(processing.compute_shear_index(norm), mode="cap")
        ice = _cap_non_finite(processing.compute_inverse_consistency_error(norm), mode="zero")

        payload = _scalar_field_payload(det, slice_index, axis)
        payload["frac_negative"] = frac_neg
        payload["log_jacobian"] = log_jac
        payload["shear"] = _scalar_field_payload(shear, slice_index, axis)
        payload["inverse_consistency"] = _scalar_field_payload(ice, slice_index, axis)

        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/compute-metrics")
async def compute_metrics(
    fixed: UploadFile | None = File(None),
    moving: UploadFile | None = File(None),
    fixed_seg: UploadFile | None = File(None),
    moving_seg: UploadFile | None = File(None),
    disp_field: UploadFile | None = None,
):
    try:
        if fixed_seg is not None and moving_seg is not None:
            fdata = await fixed_seg.read()
            mdata = await moving_seg.read()
            fixed_name = fixed_seg.filename
            moving_name = moving_seg.filename
        elif fixed is not None and moving is not None:
            fdata = await fixed.read()
            mdata = await moving.read()
            fixed_name = fixed.filename
            moving_name = moving.filename
        else:
            raise HTTPException(
                status_code=400,
                detail="Fixed and moving segmentation files are required for metric computation.",
            )

        A = _load_image_bytes(fdata, fixed_name)
        B = _load_image_bytes(mdata, moving_name)
        dice = processing.dice_score(A, B)
        dice95 = processing.dice95(A, B)
        hd = processing.hausdorff(A, B)
        hd95 = processing.hausdorff95(A, B)
        result = {
            "dice": dice,
            "dice95": dice95,
            "hausdorff": hd,
            "hausdorff95": hd95,
            "per_structure": processing.per_label_metrics(A, B),
        }

        if disp_field is not None:
            ddata = await disp_field.read()
            disp = _load_image_bytes(ddata, disp_field.filename)
            norm = processing.normalize_disp(disp)
            det = processing.compute_jacobian_det(norm)
            result["frac_negative_jacobian"] = processing.frac_negative_jacobian(det)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/generate-samples")
async def generate_samples():
    os.makedirs(sample_dir, exist_ok=True)
    test_data.generate_and_save_samples(sample_dir)
    files = [f for f in os.listdir(sample_dir) if os.path.isfile(os.path.join(sample_dir, f))]
    urls = [f"http://localhost:8000/test_samples/{f}" for f in sorted(files)]
    return JSONResponse({"generated_files": sorted(files), "urls": urls})


@app.post("/run-registration")
async def run_registration(
    fixed: UploadFile = File(...),
    moving: UploadFile = File(...),
    moving_seg: UploadFile | None = File(None),
    registration_type: str = Form("deformable"),
    bending_energy: float | None = Form(None),
    max_iterations: int | None = Form(None),
    grid_spacing: float | None = Form(None),
):
    try:
        job_id, job_dir = registration.create_job(
            registration_jobs_dir, registration_type, bending_energy, max_iterations, grid_spacing
        )
        fixed_bytes = await fixed.read()
        moving_bytes = await moving.read()
        moving_seg_bytes = await moving_seg.read() if moving_seg is not None else None
        registration.start_job(
            job_id,
            job_dir,
            fixed_bytes,
            fixed.filename,
            moving_bytes,
            moving.filename,
            moving_seg_bytes,
            moving_seg.filename if moving_seg is not None else None,
        )
        return JSONResponse({"job_id": job_id})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/registration-status/{job_id}")
async def registration_status(job_id: str):
    status = registration.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Unknown job id")

    results = status["results"]
    if results:
        results = {
            "warped_image": f"http://localhost:8000/registration_jobs/{job_id}/{results['warped_image']}",
            "displacement": f"http://localhost:8000/registration_jobs/{job_id}/{results['displacement']}",
            "warped_segmentation": (
                f"http://localhost:8000/registration_jobs/{job_id}/{results['warped_segmentation']}"
                if results["warped_segmentation"]
                else None
            ),
        }

    return JSONResponse({
        "status": status["status"],
        "phase": status["phase"],
        "progress": status["progress"],
        "error": status["error"],
        "results": results,
    })


if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
