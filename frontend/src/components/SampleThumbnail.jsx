import { useEffect, useRef, useState } from "react";

// Tiny read-only preview for a sample-picker card: fetches the sample file,
// asks the backend to normalize a slice, and paints it into a small canvas.
// No slice control, no overlay - just enough to recognize the image at a glance.
function SampleThumbnail({ url, label }) {
  const canvasRef = useRef(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let canceled = false;
    setFailed(false);

    const previewSlice = (file, sliceIndex) => {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("slice_index", String(sliceIndex));
      formData.append("mode", "image");
      return fetch("http://localhost:8000/preview-slice", { method: "POST", body: formData }).then((response) => {
        if (!response.ok) throw new Error("Unable to load thumbnail");
        return response.json();
      });
    };

    fetch(url)
      .then((response) => response.blob())
      .then((blob) => {
        const file = new File([blob], url.split("/").pop());
        return previewSlice(file, 0).then((first) => {
          // For a 3D volume, slice 0 is often near-empty background - use a
          // representative middle slice instead so the thumbnail is legible.
          if (first?.is_3d && (first.slice_count || 1) > 1) {
            return previewSlice(file, Math.floor((first.slice_count - 1) / 2));
          }
          return first;
        });
      })
      .then((json) => {
        if (canceled || !json?.data || !canvasRef.current) return;
        const rows = json.data.length;
        const cols = rows > 0 ? json.data[0].length : 0;
        if (!rows || !cols) return;

        const canvas = canvasRef.current;
        canvas.width = cols;
        canvas.height = rows;
        const context = canvas.getContext("2d");
        const imageData = context.createImageData(cols, rows);
        const pixels = imageData.data;

        for (let row = 0; row < rows; row += 1) {
          for (let col = 0; col < cols; col += 1) {
            const value = Math.max(0, Math.min(255, json.data[row][col]));
            const index = (row * cols + col) * 4;
            pixels[index] = value;
            pixels[index + 1] = value;
            pixels[index + 2] = value;
            pixels[index + 3] = 255;
          }
        }
        context.putImageData(imageData, 0, 0);
      })
      .catch((error) => {
        console.error(error);
        if (!canceled) setFailed(true);
      });

    return () => {
      canceled = true;
    };
  }, [url]);

  return (
    <div className="sample-thumb">
      <div className="sample-thumb-frame">
        {failed ? <span className="sample-thumb-fallback">?</span> : <canvas ref={canvasRef} className="sample-thumb-canvas" />}
      </div>
      <span className="sample-thumb-label">{label}</span>
    </div>
  );
}

export default SampleThumbnail;
