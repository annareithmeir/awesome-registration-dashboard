import os
import numpy as np

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from . import processing, test_data

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


def _load_image_bytes(data: bytes, filename: str):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        return processing.load_nifti_bytes(data)
    if filename.endswith(".pt"):
        return processing.load_pt_bytes(data)
    raise ValueError("Unsupported file type: %s" % filename)


@app.post("/preview-slice")
async def preview_slice(file: UploadFile = File(...), slice_index: int = Form(0), mode: str = Form("image")):
    try:
        data = await file.read()
        arr = _load_image_bytes(data, file.filename)
        arr = np.asarray(arr, dtype=np.float32)

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
                if slice_index < 0 or slice_index >= arr.shape[0]:
                    raise HTTPException(status_code=400, detail="Slice index out of range")
                disp = arr[slice_index]
                yy, xx = np.mgrid[0:disp.shape[0], 0:disp.shape[1]]
                warp_x = xx.astype(np.float32) + disp[..., 0]
                warp_y = yy.astype(np.float32) + disp[..., 1]
                preview = np.zeros((disp.shape[0], disp.shape[1]), dtype=np.float32)
                volume_shape = arr.shape
                slice_count = arr.shape[0]
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
                if slice_index < 0 or slice_index >= arr.shape[0]:
                    raise HTTPException(status_code=400, detail="Slice index out of range")
                preview = arr[slice_index]
                volume_shape = arr.shape
                slice_count = arr.shape[0]
                is_3d = True
            elif arr.ndim == 4:
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


@app.post("/compute-jacobian")
async def compute_jacobian(disp_field: UploadFile = File(...), slice_index: int = Form(0)):
    try:
        data = await disp_field.read()
        arr = _load_image_bytes(data, disp_field.filename)
        norm = processing.normalize_disp(arr)
        det = processing.compute_jacobian_det(norm)
        frac_neg = processing.frac_negative_jacobian(det)

        if det.ndim == 2:
            preview = det
            volume_shape = det.shape
            slice_count = 1
            is_3d = False
        elif det.ndim == 3:
            if slice_index < 0 or slice_index >= det.shape[0]:
                raise HTTPException(status_code=400, detail="Slice index out of range")
            preview = det[slice_index]
            volume_shape = det.shape
            slice_count = det.shape[0]
            is_3d = True
        else:
            raise HTTPException(status_code=400, detail="Unsupported Jacobian determinant dimensions")

        preview = np.nan_to_num(preview, nan=0.0, posinf=0.0, neginf=0.0)
        min_val = float(np.min(preview))
        max_val = float(np.max(preview))
        if np.isclose(max_val, min_val):
            normalized = np.zeros_like(preview, dtype=np.uint8)
        else:
            normalized = ((preview - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

        return JSONResponse({
            "min": float(np.min(det)),
            "max": float(np.max(det)),
            "mean": float(np.mean(det)),
            "frac_negative": frac_neg,
            "shape": det.shape,
            "preview_shape": list(preview.shape),
            "volume_shape": list(volume_shape),
            "slice_count": slice_count,
            "is_3d": is_3d,
            "data": normalized.tolist(),
            "values": preview.tolist(),
        })
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


if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
