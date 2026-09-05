import { useEffect, useRef, useState } from "react";
import { rotateGrid } from "../utils/rotateGrid";

function JacobianViewer({ title, jacobian, rotation = 0 }) {
  const canvasRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [showTopology, setShowTopology] = useState(false);

  useEffect(() => {
    setPreview(jacobian?.data ? jacobian : null);
  }, [jacobian]);

  useEffect(() => {
    if (!canvasRef.current || !preview?.data) return;
    const canvas = canvasRef.current;
    const previewData = rotateGrid(preview.data, rotation);
    const rows = previewData.length;
    const cols = rows > 0 ? previewData[0].length : 0;
    if (!rows || !cols) return;

    canvas.width = cols;
    canvas.height = rows;
    const context = canvas.getContext("2d");
    const imageData = context.createImageData(cols, rows);
    const pixels = imageData.data;

    const detValues = preview.values ? rotateGrid(preview.values, rotation) : null;
    const hasDetValues = Array.isArray(detValues) && detValues.length > 0;
    const minDet = Number.isFinite(preview.min) ? preview.min : null;
    const maxDet = Number.isFinite(preview.max) ? preview.max : null;
    const useTopology = showTopology;
    // J <= 0: folding (non-invertible / orientation-reversing).
    const foldThreshold = 1e-6;
    // |J - 1| within this band counts as (approximately) volume-preserving.
    const preservationBand = 0.02;
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
            const normalized = Math.max(0, Math.min(255, previewData[row][col]));
            value = valueFromNormalized(normalized);
          }

          if (value <= foldThreshold) {
            // Folding: J <= 0
            pixels[index] = 222;
            pixels[index + 1] = 82;
            pixels[index + 2] = 75;
          } else if (Math.abs(value - 1) <= preservationBand) {
            // Preservation: J == 1 (within a small tolerance)
            pixels[index] = 94;
            pixels[index + 1] = 147;
            pixels[index + 2] = 236;
          } else if (value < 1) {
            // Shrinkage: 0 < J < 1
            pixels[index] = 240;
            pixels[index + 1] = 164;
            pixels[index + 2] = 66;
          } else {
            // Growth: J > 1
            pixels[index] = 88;
            pixels[index + 1] = 196;
            pixels[index + 2] = 113;
          }
          pixels[index + 3] = 255;
          continue;
        }

        const value = previewData[row][col];
        const normalized = Math.max(0, Math.min(255, value));
        pixels[index] = normalized;
        pixels[index + 1] = normalized;
        pixels[index + 2] = normalized;
        pixels[index + 3] = 255;
      }
    }

    context.putImageData(imageData, 0, 0);
  }, [preview, showTopology, rotation]);

  return (
    <div className="viewer-card jacobian-viewer">
      <div className="viewer-header">
        <h3>{title}</h3>
        <div className="viewer-controls">
          <label className="toggle-field">
            <span className="toggle-field-label">Topology changes</span>
            <span className="toggle-switch">
              <input type="checkbox" checked={showTopology} onChange={(e) => setShowTopology(e.target.checked)} />
              <span className="toggle-slider" />
            </span>
          </label>
        </div>
      </div>
      <div className="viewer-box">
        {/* The legend overlays the preview instead of sitting above it, so
            toggling it (or the plain Jacobian tile having no legend at all)
            never changes the preview box's size - every viewer tile keeps
            an identical-size preview area, and the images all line up. */}
        <div className="viewer-preview">
          {preview ? <canvas ref={canvasRef} className="preview-canvas" /> : "No jacobian preview yet"}
          <div className={`jacobian-legend jacobian-legend-overlay ${showTopology ? "" : "jacobian-legend-hidden"}`}>
            <div className="legend-item">
              <span className="legend-swatch legend-negative" />
              Folding
            </div>
            <div className="legend-item">
              <span className="legend-swatch legend-shrinkage" />
              Shrinkage
            </div>
            <div className="legend-item">
              <span className="legend-swatch legend-zero" />
              Preservation
            </div>
            <div className="legend-item">
              <span className="legend-swatch legend-positive" />
              Growth
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default JacobianViewer;
