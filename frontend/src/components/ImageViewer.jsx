import { useEffect, useRef, useState } from "react";

function ImageViewer({
  title,
  file,
  overlayFile,
  sliceIndex,
  onSliceChange,
  mode = "image",
  overlayEnabled = false,
  overlayOpacity = 0.65,
  onPreviewInfoChange,
}) {
  const [preview, setPreview] = useState(null);
  const [overlayPreview, setOverlayPreview] = useState(null);
  const [shape, setShape] = useState(null);
  const [is3D, setIs3D] = useState(false);
  const [isDisplacement, setIsDisplacement] = useState(false);
  const [dispRenderMode, setDispRenderMode] = useState("grid");
  const [maxSlice, setMaxSlice] = useState(0);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      setShape(null);
      setIs3D(false);
      setMaxSlice(0);
      return;
    }

    const isNifti = file.name.endsWith(".nii") || file.name.endsWith(".nii.gz");
    if (!isNifti && !file.name.endsWith(".pt")) {
      setPreview(null);
      setShape(null);
      setIs3D(false);
      setMaxSlice(0);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("slice_index", String(sliceIndex));
    formData.append("mode", mode);

    fetch("http://localhost:8000/preview-slice", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load preview");
        return response.json();
      })
      .then((json) => {
        setPreview(json);
        setShape(json.volume_shape || json.shape || []);
        setIs3D(Boolean(json.is_3d));
        onPreviewInfoChange?.({ is3D: Boolean(json.is_3d) });
        setIsDisplacement(file.name.endsWith(".pt"));
        setMaxSlice(Math.max(0, (json.slice_count || 1) - 1));
      })
      .catch((error) => {
        console.error(error);
        setPreview(null);
        onPreviewInfoChange?.({ is3D: false });
      });
  }, [file, sliceIndex, mode]);

  useEffect(() => {
    if (!overlayFile) {
      setOverlayPreview(null);
      return;
    }

    const formData = new FormData();
    formData.append("file", overlayFile);
    formData.append("slice_index", String(sliceIndex));
    formData.append("mode", "segmentation");

    fetch("http://localhost:8000/preview-slice", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load overlay preview");
        return response.json();
      })
      .then((json) => {
        setOverlayPreview(json);
      })
      .catch((error) => {
        console.error(error);
        setOverlayPreview(null);
      });
  }, [overlayFile, sliceIndex]);

  useEffect(() => {
    if (!preview?.data) return;
    const canvas = canvasRef.current;
    const rows = preview.data.length;
    const cols = rows > 0 ? preview.data[0].length : 0;
    if (!rows || !cols) return;

    canvas.width = cols;
    canvas.height = rows;
    const context = canvas.getContext("2d");

    if (isDisplacement && preview.warp_x && preview.warp_y) {
      if (dispRenderMode === "grid") {
        const displayWidth = canvas.clientWidth || cols;
        const displayHeight = canvas.clientHeight || rows;
        const renderScale = Math.max(
          2,
          Math.min(8, Math.ceil(Math.min(displayWidth / cols, displayHeight / rows)))
        );

        canvas.width = Math.max(1, Math.round(cols * renderScale));
        canvas.height = Math.max(1, Math.round(rows * renderScale));
        context.setTransform(renderScale, 0, 0, renderScale, 0, 0);
        context.imageSmoothingEnabled = false;
        context.clearRect(0, 0, cols, rows);
        context.strokeStyle = "rgba(255,255,255,0.82)";
        context.lineWidth = 0.18;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.setLineDash([]);

        const warpX = preview.warp_x;
        const warpY = preview.warp_y;
        const gridStep = Math.max(1, Math.floor(Math.min(rows, cols) / 32));

        for (let y = 0; y < rows; y += gridStep) {
          context.beginPath();
          context.moveTo(warpX[y][0], warpY[y][0]);
          for (let x = 1; x < cols; x += 1) {
            context.lineTo(warpX[y][x], warpY[y][x]);
          }
          context.stroke();
        }

        for (let x = 0; x < cols; x += gridStep) {
          context.beginPath();
          context.moveTo(warpX[0][x], warpY[0][x]);
          for (let y = 1; y < rows; y += 1) {
            context.lineTo(warpX[y][x], warpY[y][x]);
          }
          context.stroke();
        }

        return;
      }

      if (dispRenderMode === "color") {
        context.clearRect(0, 0, cols, rows);
        const imageData = context.createImageData(cols, rows);
        const pixels = imageData.data;

        const getHue = (dx, dy) => {
          const angle = Math.atan2(dy, dx);
          return ((angle + Math.PI) / (2 * Math.PI)) * 360;
        };

        const hsvToRgb = (h, s, v) => {
          const c = v * s;
          const hp = h / 60;
          const x = c * (1 - Math.abs((hp % 2) - 1));
          let r = 0,
            g = 0,
            b = 0;
          if (hp >= 0 && hp < 1) {
            r = c;
            g = x;
          } else if (hp >= 1 && hp < 2) {
            r = x;
            g = c;
          } else if (hp >= 2 && hp < 3) {
            g = c;
            b = x;
          } else if (hp >= 3 && hp < 4) {
            g = x;
            b = c;
          } else if (hp >= 4 && hp < 5) {
            r = x;
            b = c;
          } else {
            r = c;
            b = x;
          }
          const m = v - c;
          return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
        };

        let maxMag = 0;
        for (let row = 0; row < rows; row += 1) {
          for (let col = 0; col < cols; col += 1) {
            const dx = preview.warp_x[row][col] - col;
            const dy = preview.warp_y[row][col] - row;
            const mag = Math.hypot(dx, dy);
            if (mag > maxMag) maxMag = mag;
          }
        }

        for (let row = 0; row < rows; row += 1) {
          for (let col = 0; col < cols; col += 1) {
            const index = (row * cols + col) * 4;
            const dx = preview.warp_x[row][col] - col;
            const dy = preview.warp_y[row][col] - row;
            const mag = Math.hypot(dx, dy);
            const hue = getHue(dx, dy);
            const value = maxMag > 0 ? Math.min(1, mag / maxMag) : 0;
            const saturation = value > 0 ? 1 : 0;
            const [r, g, b] = hsvToRgb(hue, saturation, value);
            pixels[index] = r;
            pixels[index + 1] = g;
            pixels[index + 2] = b;
            pixels[index + 3] = 255;
          }
        }

        context.putImageData(imageData, 0, 0);
        return;
      }
    }

    const imageData = context.createImageData(cols, rows);
    const pixels = imageData.data;

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const value = preview.data[row][col];
        const index = (row * cols + col) * 4;

        if (overlayEnabled && overlayPreview?.data) {
          const base = Math.max(0, Math.min(255, value));
          const segValue = Math.round(Number(overlayPreview.data[row][col] ?? 0));
          if (segValue <= 0) {
            pixels[index] = 0;
            pixels[index + 1] = 0;
            pixels[index + 2] = 0;
          } else {
            const alpha = overlayOpacity;
            const [overlayR, overlayG, overlayB] = getSegmentationColor(segValue);
            pixels[index] = base * (1 - alpha) + overlayR * alpha;
            pixels[index + 1] = base * (1 - alpha) + overlayG * alpha;
            pixels[index + 2] = base * (1 - alpha) + overlayB * alpha;
          }
        } else {
          const normalized = Math.max(0, Math.min(255, value));
          pixels[index] = normalized;
          pixels[index + 1] = normalized;
          pixels[index + 2] = normalized;
        }
        pixels[index + 3] = 255;
      }
    }

    context.putImageData(imageData, 0, 0);
  }, [preview, overlayPreview, overlayEnabled, overlayOpacity, isDisplacement, dispRenderMode]);

  function hslToRgb(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
    const m = l - c / 2;

    let r = 0;
    let g = 0;
    let b = 0;

    if (h < 60) {
      r = c; g = x; b = 0;
    } else if (h < 120) {
      r = x; g = c; b = 0;
    } else if (h < 180) {
      r = 0; g = c; b = x;
    } else if (h < 240) {
      r = 0; g = x; b = c;
    } else if (h < 300) {
      r = x; g = 0; b = c;
    } else {
      r = c; g = 0; b = x;
    }

    return [r + m, g + m, b + m];
  }

  function getSegmentationColor(value) {
    const numericValue = Number(value) || 0;
    if (!numericValue) {
      return [0, 0, 0];
    }

    if (numericValue === 1) {
      return [45, 156, 219];
    }

    if (numericValue === 2) {
      return [241, 146, 32];
    }

    const hue = 300;
    const sat = 0.75;
    const light = 0.55;
    const [r, g, b] = hslToRgb(hue, sat, light);
    return [r * 255, g * 255, b * 255];
  }

  useEffect(() => {
    if (!shape || shape.length <= 2) return;
    const next = Math.min(sliceIndex, Math.max(0, shape[0] - 1));
    if (next !== sliceIndex) {
      onSliceChange(next);
    }
  }, [shape, sliceIndex, onSliceChange]);

  return (
    <div className="viewer-card">
      <div className="viewer-header">
        <h3>{title}</h3>
      </div>
      <div className="viewer-box">
        <div className="viewer-preview">
          {preview ? <canvas ref={canvasRef} className="preview-canvas" /> : "No image loaded"}
        </div>
        {isDisplacement && (
          <div className="viewer-controls">
            <label>
              Visualization:
              <select value={dispRenderMode} onChange={(e) => setDispRenderMode(e.target.value)}>
                <option value="grid">Warped grid</option>
                <option value="color">Color wheel</option>
              </select>
            </label>
            {dispRenderMode === "color" && (
              <div className="viewer-note">Hue encodes displacement direction; brightness encodes magnitude.</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ImageViewer;
