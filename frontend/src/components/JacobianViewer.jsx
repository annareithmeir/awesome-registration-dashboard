import { useEffect, useRef, useState } from "react";
import { rotateGrid } from "../utils/rotateGrid";

// J <= 0: folding (non-invertible / orientation-reversing) - shared by both
// topology color schemes below, checked first regardless of which is active.
const FOLD_THRESHOLD = 1e-6;

// The original discrete scheme: just folding / shrinking / preserved /
// growing, one flat color each.
const SIMPLE_PRESERVATION_BAND = 0.02;
const SIMPLE_TOPOLOGY_BINS = [
  { label: "Folding", color: [222, 82, 75], test: (j) => j <= FOLD_THRESHOLD },
  { label: "Preservation", color: [94, 147, 236], test: (j) => Math.abs(j - 1) <= SIMPLE_PRESERVATION_BAND },
  { label: "Shrinkage", color: [240, 164, 66], test: (j) => j < 1 },
  { label: "Growth", color: [88, 196, 113], test: () => true },
];

// The graded scheme: same four concepts, but shrinkage and growth are each
// further split into three intensity levels (light = mild, dark = severe),
// so how MUCH a region changed is visible at a glance, not just whether it
// grew or shrank. Bin edges are chosen so "a lot" and "slightly" read as
// roughly the inverse of each other in each direction (e.g. 1.25 <-> 0.8,
// 2.0 <-> 0.5) - checked in order, first match wins.
const GRADED_PRESERVATION_BAND = 0.05;
const GRADED_TOPOLOGY_BINS = [
  { label: "Folding", color: [222, 82, 75], test: (j) => j <= FOLD_THRESHOLD },
  { label: "Shrinking a lot", color: [191, 87, 0], test: (j) => j < 0.5 },
  { label: "Shrinking", color: [255, 167, 38], test: (j) => j < 0.8 },
  { label: "Shrinking slightly", color: [255, 224, 178], test: (j) => j < 1 - GRADED_PRESERVATION_BAND },
  { label: "Preserved", color: [94, 147, 236], test: (j) => j <= 1 + GRADED_PRESERVATION_BAND },
  { label: "Expanding slightly", color: [200, 230, 201], test: (j) => j <= 1.25 },
  { label: "Expanding", color: [102, 187, 106], test: (j) => j <= 2 },
  { label: "Expanding a lot", color: [27, 94, 32], test: () => true },
];

function classifyTopology(value, bins) {
  for (const bin of bins) {
    if (bin.test(value)) return bin;
  }
  return bins[bins.length - 1];
}

function JacobianViewer({ title, jacobian, rotation = 0 }) {
  const canvasRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [showTopology, setShowTopology] = useState(false);
  const [showTopologyGraded, setShowTopologyGraded] = useState(false);

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
    const topologyBins = showTopologyGraded ? GRADED_TOPOLOGY_BINS : SIMPLE_TOPOLOGY_BINS;
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

          const bin = classifyTopology(value, topologyBins);
          pixels[index] = bin.color[0];
          pixels[index + 1] = bin.color[1];
          pixels[index + 2] = bin.color[2];
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
  }, [preview, showTopology, showTopologyGraded, rotation]);

  const activeTopologyBins = showTopologyGraded ? GRADED_TOPOLOGY_BINS : SIMPLE_TOPOLOGY_BINS;

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
          {showTopology && (
            <label className="toggle-field">
              <span className="toggle-field-label">Graded intensity</span>
              <span className="toggle-switch">
                <input
                  type="checkbox"
                  checked={showTopologyGraded}
                  onChange={(e) => setShowTopologyGraded(e.target.checked)}
                />
                <span className="toggle-slider" />
              </span>
            </label>
          )}
        </div>
      </div>
      <div className="viewer-box">
        {/* The legend overlays the preview instead of sitting above it, so
            toggling it (or the plain Jacobian tile having no legend at all)
            never changes the preview box's size - every viewer tile keeps
            an identical-size preview area, and the images all line up. */}
        <div className="viewer-preview">
          {preview ? <canvas ref={canvasRef} className="preview-canvas" /> : "No jacobian preview yet"}
          <div
            className={`jacobian-legend jacobian-legend-overlay ${showTopology ? "" : "jacobian-legend-hidden"} ${showTopologyGraded ? "jacobian-legend-graded" : ""}`}
          >
            {activeTopologyBins.map((bin) => (
              <div className="legend-item" key={bin.label}>
                <span className="legend-swatch" style={{ background: `rgb(${bin.color.join(",")})` }} />
                {bin.label}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default JacobianViewer;
