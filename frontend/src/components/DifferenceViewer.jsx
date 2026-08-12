import { useEffect, useRef, useState } from "react";

function DifferenceViewer({ title, fixedFile, movingFile, fixedSegFile, movingSegFile, sliceIndex }) {
  const [preview, setPreview] = useState(null);
  const [message, setMessage] = useState("Loading difference...");
  const [useSegmentations, setUseSegmentations] = useState(false);
  const canvasRef = useRef(null);

  useEffect(() => {
    const activeFixed = useSegmentations ? fixedSegFile : fixedFile;
    const activeMoving = useSegmentations ? movingSegFile : movingFile;

    if (!activeFixed || !activeMoving) {
      setPreview(null);
      setMessage(useSegmentations ? "Fixed and moving segmentations required for difference view." : "Fixed and moving images required for difference view.");
      return;
    }

    const loadSlice = async (file) => {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("slice_index", String(sliceIndex));
      formData.append("mode", useSegmentations ? "segmentation" : "image");
      const response = await fetch("http://localhost:8000/preview-slice", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Unable to load preview");
      return response.json();
    };

    let canceled = false;
    setMessage("Loading difference...");

    Promise.all([loadSlice(activeFixed), loadSlice(activeMoving)])
      .then(([fixedPreview, movingPreview]) => {
        if (canceled) return;
        if (!fixedPreview?.data || !movingPreview?.data) {
          setPreview(null);
          setMessage("Unable to compute difference preview.");
          return;
        }
        if (
          fixedPreview.data.length !== movingPreview.data.length ||
          fixedPreview.data[0]?.length !== movingPreview.data[0]?.length
        ) {
          setPreview(null);
          setMessage("Image shapes do not match for difference view.");
          return;
        }

        const rows = fixedPreview.data.length;
        const cols = rows > 0 ? fixedPreview.data[0].length : 0;
        const diffData = Array.from({ length: rows }, (_, row) =>
          Array.from({ length: cols }, (_, col) =>
            Math.abs(fixedPreview.data[row][col] - movingPreview.data[row][col])
          )
        );
        setPreview({ data: diffData, rows, cols });
      })
      .catch((error) => {
        console.error(error);
        if (!canceled) {
          setPreview(null);
          setMessage("Unable to load difference preview.");
        }
      });

    return () => {
      canceled = true;
    };
  }, [fixedFile, movingFile, fixedSegFile, movingSegFile, sliceIndex, useSegmentations]);

  useEffect(() => {
    if (!preview?.data || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const rows = preview.data.length;
    const cols = rows > 0 ? preview.data[0].length : 0;
    if (!rows || !cols) return;

    canvas.width = cols;
    canvas.height = rows;
    const context = canvas.getContext("2d");
    const imageData = context.createImageData(cols, rows);
    const pixels = imageData.data;

    let maxDiff = 0;
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const diff = preview.data[row][col];
        if (diff > maxDiff) maxDiff = diff;
      }
    }

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const index = (row * cols + col) * 4;
        const diff = preview.data[row][col];
        const value = maxDiff > 0 ? Math.round((diff / maxDiff) * 255) : 0;
        pixels[index] = value;
        pixels[index + 1] = value;
        pixels[index + 2] = value;
        pixels[index + 3] = 255;
      }
    }

    context.putImageData(imageData, 0, 0);
  }, [preview]);

  return (
    <div className="viewer-card">
      <div className="viewer-header">
        <h3>{title}</h3>
        <div className="viewer-controls">
          {fixedSegFile && movingSegFile && (
            <label className="checkbox-control">
              <input type="checkbox" checked={useSegmentations} onChange={(e) => setUseSegmentations(e.target.checked)} />
              Segmentations
            </label>
          )}
        </div>
      </div>
      <div className="viewer-box">
        <div className="viewer-preview">
          {preview ? <canvas ref={canvasRef} className="preview-canvas" /> : message}
        </div>
      </div>
    </div>
  );
}

export default DifferenceViewer;
