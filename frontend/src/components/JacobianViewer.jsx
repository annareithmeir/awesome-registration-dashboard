import { useEffect, useRef, useState } from "react";

function JacobianViewer({ title, jacobian, sliceIndex, onSliceChange }) {
  const canvasRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [maxSlice, setMaxSlice] = useState(0);
  const [showTopology, setShowTopology] = useState(false);

  useEffect(() => {
    if (!jacobian?.data) {
      setPreview(null);
      setMaxSlice(0);
      return;
    }

    setPreview(jacobian);
    setMaxSlice(Math.max(0, (jacobian.slice_count || 1) - 1));
  }, [jacobian]);

  useEffect(() => {
    if (!canvasRef.current || !preview?.data) return;
    const canvas = canvasRef.current;
    const rows = preview.data.length;
    const cols = rows > 0 ? preview.data[0].length : 0;
    if (!rows || !cols) return;

    canvas.width = cols;
    canvas.height = rows;
    const context = canvas.getContext("2d");
    const imageData = context.createImageData(cols, rows);
    const pixels = imageData.data;

    const detValues = preview.values ?? null;
    const hasDetValues = Array.isArray(detValues) && detValues.length > 0;
    const minDet = Number.isFinite(preview.min) ? preview.min : null;
    const maxDet = Number.isFinite(preview.max) ? preview.max : null;
    const useTopology = showTopology;
    const zeroThreshold = 1e-6;
    const valueFromNormalized = (normalized) => {
      if (hasDetValues) return normalized;
      if (minDet !== null && maxDet !== null && maxDet !== minDet) {
        return minDet + (normalized / 255.0) * (maxDet - minDet);
      }
      return normalized - 128;
    };

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const index = (row * cols + col) * 4;
        if (useTopology) {
          let value;
          if (hasDetValues) {
            value = detValues[row][col];
          } else {
            const normalized = Math.max(0, Math.min(255, preview.data[row][col]));
            value = valueFromNormalized(normalized);
          }

          if (value < -zeroThreshold) {
            pixels[index] = 222;
            pixels[index + 1] = 82;
            pixels[index + 2] = 75;
          } else if (value > zeroThreshold) {
            pixels[index] = 88;
            pixels[index + 1] = 196;
            pixels[index + 2] = 113;
          } else {
            pixels[index] = 94;
            pixels[index + 1] = 147;
            pixels[index + 2] = 236;
          }
          pixels[index + 3] = 255;
          continue;
        }

        const value = preview.data[row][col];
        const normalized = Math.max(0, Math.min(255, value));
        pixels[index] = normalized;
        pixels[index + 1] = normalized;
        pixels[index + 2] = normalized;
        pixels[index + 3] = 255;
      }
    }

    context.putImageData(imageData, 0, 0);
  }, [preview, showTopology]);

  return (
    <div className="viewer-card jacobian-viewer">
      <div className="viewer-header">
        <h3>{title}</h3>
        <div className="viewer-controls">
          <label className="checkbox-control">
            <input type="checkbox" checked={showTopology} onChange={(e) => setShowTopology(e.target.checked)} />
            Topology changes
          </label>
        </div>
      </div>
      <div className="viewer-box">
        <div className="viewer-preview">
          {preview ? <canvas ref={canvasRef} className="preview-canvas" /> : "No jacobian preview yet"}
        </div>
        <div className={`jacobian-legend ${showTopology ? "" : "jacobian-legend-hidden"}`}>
            <div className="legend-item">
              <span className="legend-swatch legend-negative" />
              Negative determinant
            </div>
            <div className="legend-item">
              <span className="legend-swatch legend-zero" />
              Zero determinant
            </div>
            <div className="legend-item">
              <span className="legend-swatch legend-positive" />
              Positive determinant
            </div>
        </div>
        {preview?.is_3d && (
          <div className="slice-control">
            <label>
              Slice: {Math.min(sliceIndex, maxSlice) + 1} / {maxSlice + 1}
              <input
                type="range"
                min="0"
                max={maxSlice}
                value={Math.min(sliceIndex, maxSlice)}
                onChange={(e) => onSliceChange(Number(e.target.value))}
              />
            </label>
          </div>
        )}
      </div>
    </div>
  );
}

export default JacobianViewer;
