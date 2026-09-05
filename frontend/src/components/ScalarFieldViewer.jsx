import { useEffect, useRef, useState } from "react";
import { rotateGrid } from "../utils/rotateGrid";

// Generic heatmap viewer for a per-voxel scalar field returned alongside the
// Jacobian determinant (shear index, inverse consistency error, ...). Maps
// the field's own min/max range onto a two-color ramp so each field gets its
// own reading regardless of scale.
function ScalarFieldViewer({
  title,
  field,
  lowColor = [45, 156, 219],
  highColor = [222, 82, 75],
  rotation = 0,
}) {
  const canvasRef = useRef(null);
  const [preview, setPreview] = useState(null);

  useEffect(() => {
    setPreview(field?.data ? field : null);
  }, [field]);

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
    const [lr, lg, lb] = lowColor;
    const [hr, hg, hb] = highColor;

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const index = (row * cols + col) * 4;
        const t = Math.max(0, Math.min(255, previewData[row][col])) / 255;
        pixels[index] = lr + (hr - lr) * t;
        pixels[index + 1] = lg + (hg - lg) * t;
        pixels[index + 2] = lb + (hb - lb) * t;
        pixels[index + 3] = 255;
      }
    }

    context.putImageData(imageData, 0, 0);
  }, [preview, lowColor, highColor, rotation]);

  return (
    <div className="viewer-card jacobian-viewer">
      <div className="viewer-header">
        <h3>{title}</h3>
      </div>
      <div className="viewer-box">
        <div className="viewer-preview">
          {preview ? <canvas ref={canvasRef} className="preview-canvas" /> : "No data yet"}
        </div>
      </div>
    </div>
  );
}

export default ScalarFieldViewer;
