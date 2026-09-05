import { useEffect, useRef, useState } from "react";
import { getSegmentationColor } from "../utils/segmentationColors";
import { rotateGrid, rotateWarpGrid } from "../utils/rotateGrid";

function ImageViewer({
  title,
  file,
  overlayFile,
  backgroundFile,
  sliceIndex,
  onSliceChange,
  axis = 0,
  mode = "image",
  overlayEnabled = false,
  overlayOpacity = 0.65,
  onOverlayEnabledChange,
  onPreviewInfoChange,
  rotation = 0,
}) {
  const [preview, setPreview] = useState(null);
  const [overlayPreview, setOverlayPreview] = useState(null);
  const [backgroundPreview, setBackgroundPreview] = useState(null);
  const [shape, setShape] = useState(null);
  const [is3D, setIs3D] = useState(false);
  const [isDisplacement, setIsDisplacement] = useState(false);
  const [showBackground, setShowBackground] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [showColorWheel, setShowColorWheel] = useState(false);
  const [maxSlice, setMaxSlice] = useState(0);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      setShape(null);
      setIs3D(false);
      setMaxSlice(0);
      setLoading(false);
      return;
    }

    const isNifti = file.name.endsWith(".nii") || file.name.endsWith(".nii.gz");
    if (!isNifti && !file.name.endsWith(".pt")) {
      setPreview(null);
      setShape(null);
      setIs3D(false);
      setMaxSlice(0);
      setLoading(false);
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("slice_index", String(sliceIndex));
    formData.append("mode", mode);
    formData.append("axis", String(axis));

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
        onPreviewInfoChange?.({
          is3D: Boolean(json.is_3d),
          sliceCount: json.slice_count || 1,
          shape: json.volume_shape || json.shape || [],
        });
        setIsDisplacement(file.name.endsWith(".pt"));
        setMaxSlice(Math.max(0, (json.slice_count || 1) - 1));
      })
      .catch((error) => {
        console.error(error);
        setPreview(null);
        onPreviewInfoChange?.({ is3D: false, sliceCount: 1, shape: [] });
      })
      .finally(() => setLoading(false));
  }, [file, sliceIndex, axis, mode]);

  useEffect(() => {
    if (!overlayFile) {
      setOverlayPreview(null);
      return;
    }

    const formData = new FormData();
    formData.append("file", overlayFile);
    formData.append("slice_index", String(sliceIndex));
    formData.append("mode", "segmentation");
    formData.append("axis", String(axis));

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
  }, [overlayFile, sliceIndex, axis]);

  useEffect(() => {
    if (!backgroundFile) {
      setBackgroundPreview(null);
      return;
    }

    const formData = new FormData();
    formData.append("file", backgroundFile);
    formData.append("slice_index", String(sliceIndex));
    formData.append("mode", "image");
    formData.append("axis", String(axis));

    fetch("http://localhost:8000/preview-slice", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) throw new Error("Unable to load background preview");
        return response.json();
      })
      .then((json) => {
        setBackgroundPreview(json);
      })
      .catch((error) => {
        console.error(error);
        setBackgroundPreview(null);
      });
  }, [backgroundFile, sliceIndex, axis]);

  useEffect(() => {
    if (!preview?.data) return;
    const canvas = canvasRef.current;
    const previewData = rotateGrid(preview.data, rotation);
    const overlayData = overlayPreview?.data ? rotateGrid(overlayPreview.data, rotation) : null;
    const backgroundData = backgroundPreview?.data ? rotateGrid(backgroundPreview.data, rotation) : null;
    const { warpX, warpY } =
      preview.warp_x && preview.warp_y
        ? rotateWarpGrid(preview.warp_x, preview.warp_y, rotation)
        : { warpX: null, warpY: null };
    const rows = previewData.length;
    const cols = rows > 0 ? previewData[0].length : 0;
    if (!rows || !cols) return;

    canvas.width = cols;
    canvas.height = rows;
    const context = canvas.getContext("2d");

    if (isDisplacement && warpX && warpY) {
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

      const hasBackground = showBackground && backgroundData;

      // Build the raster layer (underlying fixed image and/or color-wheel
      // magnitude/direction map) at native resolution, then upscale it onto
      // the on-screen canvas so the warped-grid overlay can be drawn crisply
      // in the same coordinate space.
      const off = document.createElement("canvas");
      off.width = cols;
      off.height = rows;
      const offContext = off.getContext("2d");
      const imageData = offContext.createImageData(cols, rows);
      const pixels = imageData.data;

      let maxMag = 0;
      if (showColorWheel) {
        for (let row = 0; row < rows; row += 1) {
          for (let col = 0; col < cols; col += 1) {
            const dx = warpX[row][col] - col;
            const dy = warpY[row][col] - row;
            const mag = Math.hypot(dx, dy);
            if (mag > maxMag) maxMag = mag;
          }
        }
      }

      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          const index = (row * cols + col) * 4;
          const base = hasBackground ? Math.max(0, Math.min(255, backgroundData[row][col])) : null;

          if (showColorWheel) {
            const dx = warpX[row][col] - col;
            const dy = warpY[row][col] - row;
            const mag = Math.hypot(dx, dy);
            const hue = getHue(dx, dy);
            const value = maxMag > 0 ? Math.min(1, mag / maxMag) : 0;
            const saturation = value > 0 ? 1 : 0;
            const [r, g, b] = hsvToRgb(hue, saturation, value);
            if (base !== null) {
              const blend = 0.75;
              pixels[index] = base * (1 - blend) + r * blend;
              pixels[index + 1] = base * (1 - blend) + g * blend;
              pixels[index + 2] = base * (1 - blend) + b * blend;
            } else {
              pixels[index] = r;
              pixels[index + 1] = g;
              pixels[index + 2] = b;
            }
            pixels[index + 3] = 255;
          } else if (base !== null) {
            pixels[index] = base;
            pixels[index + 1] = base;
            pixels[index + 2] = base;
            pixels[index + 3] = 255;
          } else {
            pixels[index + 3] = 0;
          }
        }
      }
      offContext.putImageData(imageData, 0, 0);

      const displayWidth = canvas.clientWidth || cols;
      const displayHeight = canvas.clientHeight || rows;
      const renderScale = Math.max(
        2,
        Math.min(8, Math.ceil(Math.min(displayWidth / cols, displayHeight / rows)))
      );

      canvas.width = Math.max(1, Math.round(cols * renderScale));
      canvas.height = Math.max(1, Math.round(rows * renderScale));
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.imageSmoothingEnabled = false;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(off, 0, 0, canvas.width, canvas.height);

      if (showGrid) {
        context.setTransform(renderScale, 0, 0, renderScale, 0, 0);
        context.strokeStyle = "rgba(255,255,255,0.82)";
        context.lineWidth = 0.18;
        context.lineCap = "round";
        context.lineJoin = "round";
        context.setLineDash([]);

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
        context.setTransform(1, 0, 0, 1, 0, 0);
      }

      return;
    }

    const imageData = context.createImageData(cols, rows);
    const pixels = imageData.data;

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const value = previewData[row][col];
        const index = (row * cols + col) * 4;

        if (overlayEnabled && overlayData) {
          const base = Math.max(0, Math.min(255, value));
          const segValue = Math.round(Number(overlayData[row][col] ?? 0));
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
  }, [
    preview,
    overlayPreview,
    overlayEnabled,
    overlayOpacity,
    isDisplacement,
    showBackground,
    showGrid,
    showColorWheel,
    backgroundPreview,
    rotation,
  ]);

  useEffect(() => {
    if (!shape || shape.length <= 2) return;
    const sliceDim = Math.min(Math.max(axis, 0), 2);
    const next = Math.min(sliceIndex, Math.max(0, shape[sliceDim] - 1));
    if (next !== sliceIndex) {
      onSliceChange(next);
    }
  }, [shape, sliceIndex, axis, onSliceChange]);

  return (
    <div className="viewer-card">
      <div className="viewer-header">
        <h3>{title}</h3>
        {(isDisplacement || overlayFile) && (
          <div className="viewer-controls">
            {isDisplacement && (
              <>
                <label className="toggle-field">
                  <span className="toggle-field-label">Fixed image</span>
                  <span className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={showBackground}
                      onChange={(e) => setShowBackground(e.target.checked)}
                      disabled={!backgroundFile}
                    />
                    <span className="toggle-slider" />
                  </span>
                </label>
                <label className="toggle-field">
                  <span className="toggle-field-label">Warped grid</span>
                  <span className="toggle-switch">
                    <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
                    <span className="toggle-slider" />
                  </span>
                </label>
                <label className="toggle-field">
                  <span className="toggle-field-label">Color wheel</span>
                  <span className="toggle-switch">
                    <input type="checkbox" checked={showColorWheel} onChange={(e) => setShowColorWheel(e.target.checked)} />
                    <span className="toggle-slider" />
                  </span>
                </label>
              </>
            )}
            {overlayFile && (
              <div className="toggle-switch-control">
                <span className={!overlayEnabled ? "toggle-label active" : "toggle-label"}>Image</span>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={overlayEnabled}
                    onChange={(e) => onOverlayEnabledChange?.(e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
                <span className={overlayEnabled ? "toggle-label active" : "toggle-label"}>Segmentation</span>
              </div>
            )}
          </div>
        )}
      </div>
      <div className="viewer-box">
        {/* The color-wheel note overlays the preview instead of sitting
            below it, so toggling it on/off never resizes the preview box -
            every viewer tile keeps an identical-size preview area, and the
            images all line up. */}
        <div className="viewer-preview">
          {preview ? (
            <canvas ref={canvasRef} className="preview-canvas" />
          ) : loading ? (
            <span className="viewer-loading">
              <span className="viewer-spinner" />
              Loading…
            </span>
          ) : (
            "No image loaded"
          )}
          {isDisplacement && showColorWheel && (
            <div className="viewer-note viewer-note-overlay">
              Hue encodes displacement direction; brightness encodes magnitude.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ImageViewer;
