import { useState } from "react";
import ImageViewer from "./components/ImageViewer";
import DifferenceViewer from "./components/DifferenceViewer";
import MetricPanel from "./components/MetricPanel";
import JacobianViewer from "./components/JacobianViewer";
import "./styles.css";

function App() {
  const [fixedFile, setFixedFile] = useState(null);
  const [movingFile, setMovingFile] = useState(null);
  const [fixedSegFile, setFixedSegFile] = useState(null);
  const [movingSegFile, setMovingSegFile] = useState(null);
  const [dispFile, setDispFile] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [jacobian, setJacobian] = useState(null);
  const [jacobianPreview, setJacobianPreview] = useState(null);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [globalSliceIndex, setGlobalSliceIndex] = useState(0);
  const [sampleFiles, setSampleFiles] = useState([]);
  const [sampleMessage, setSampleMessage] = useState(null);
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.65);
  const [fixedIs3D, setFixedIs3D] = useState(false);
  const [movingIs3D, setMovingIs3D] = useState(false);
  const [actionMenuOpen, setActionMenuOpen] = useState(false);
  const canUseSliceSlider = fixedIs3D && movingIs3D;
  const canComputeMetrics = Boolean(fixedSegFile && movingSegFile);
  const canComputeJacobian = Boolean(dispFile);

  const handleSliceChange = (value) => {
    setSliceIndex(value);
    setGlobalSliceIndex(value);
  };

  const handleComputeMetrics = async () => {
    const fixedSeg = fixedSegFile || fixedFile;
    const movingSeg = movingSegFile || movingFile;
    if (!fixedSeg || !movingSeg) return;
    const formData = new FormData();
    if (fixedSegFile && movingSegFile) {
      formData.append("fixed_seg", fixedSegFile);
      formData.append("moving_seg", movingSegFile);
    } else {
      formData.append("fixed", fixedSeg);
      formData.append("moving", movingSeg);
    }
    const response = await fetch("http://localhost:8000/compute-metrics", {
      method: "POST",
      body: formData,
    });
    const json = await response.json();
    setMetrics(json);
  };

  const handleComputeJacobian = async () => {
    if (!dispFile) return;
    const formData = new FormData();
    formData.append("disp_field", dispFile);
    formData.append("slice_index", String(globalSliceIndex));
    const response = await fetch("http://localhost:8000/compute-jacobian", {
      method: "POST",
      body: formData,
    });
    const json = await response.json();
    setJacobian(json);
    setJacobianPreview(json);
  };

  const loadSampleFile = async (url) => {
    const response = await fetch(url);
    const blob = await response.blob();
    const name = url.split("/").pop();
    return new File([blob], name);
  };

  const handleSetSample = async (url, setter) => {
    try {
      const file = await loadSampleFile(url);
      setter(file);
    } catch (error) {
      console.error("Sample load failed", error);
    }
  };

  const handleGenerateSamples = async () => {
    const response = await fetch("http://localhost:8000/generate-samples", {
      method: "POST",
    });
    const json = await response.json();
    const files = (json.urls || []).map((url) => ({ url, name: url.split("/").pop() }));
    setSampleFiles(files);
    setSampleMessage(`Generated ${json.generated_files.length} sample files.`);
  };

  return (
    <div className="app-shell">
      <header>
        <h1>Medical Image Registration Viewer</h1>
        <p>Load 2D/3D images, displacement fields, and compute registration metrics.</p>
      </header>

      <section className="upload-row">
        <div className="file-card-group">
          <label className="file-card">
            Fixed image
            <input type="file" onChange={(e) => setFixedFile(e.target.files?.[0] ?? null)} />
          </label>
          <label className="file-card">
            Fixed segmentation
            <input type="file" onChange={(e) => setFixedSegFile(e.target.files?.[0] ?? null)} />
          </label>
        </div>
        <div className="file-card-group">
          <label className="file-card">
            Moving image
            <input type="file" onChange={(e) => setMovingFile(e.target.files?.[0] ?? null)} />
          </label>
          <label className="file-card">
            Moving segmentation
            <input type="file" onChange={(e) => setMovingSegFile(e.target.files?.[0] ?? null)} />
          </label>
        </div>
        <label className="file-card">
          Displacement field (.pt)
          <input type="file" onChange={(e) => setDispFile(e.target.files?.[0] ?? null)} />
        </label>
      </section>

      {(sampleMessage || sampleFiles.length > 0) && (
        <section className="button-panel sample-panel">
        {sampleMessage && <div className="sample-message">{sampleMessage}</div>}
        {sampleFiles.length > 0 && (
          <div className="sample-links">
            <h3>Sample files</h3>
            <ul>
              {sampleFiles.map((file) => (
                <li key={file.url}>
                  <button className="link-button" onClick={() => handleSetSample(file.url, setFixedFile)}>
                    Set fixed: {file.name}
                  </button>
                  <button className="link-button" onClick={() => handleSetSample(file.url, setMovingFile)}>
                    Set moving: {file.name}
                  </button>
                  <button className="link-button" onClick={() => handleSetSample(file.url, setDispFile)}>
                    Set displacement: {file.name}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
        </section>
      )}

      <section className="metrics-row">
        <MetricPanel metrics={metrics} jacobian={jacobian} />
      </section>

      <section className="global-slice-control">
        <label>
          Global slice: {globalSliceIndex + 1}
          <input
            type="range"
            min="0"
            max={Math.max(0, 15)}
            value={globalSliceIndex}
            disabled={!canUseSliceSlider}
            onChange={(e) => handleSliceChange(Number(e.target.value))}
          />
        </label>
      </section>

      <section className="overlay-control-bar">
        <label className="checkbox-control">
          <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
          Show segmentation overlays
        </label>
        <label>
          Overlay opacity: {Math.round(overlayOpacity * 100)}%
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={overlayOpacity}
            disabled={!showOverlay}
            onChange={(e) => setOverlayOpacity(Number(e.target.value))}
          />
        </label>
      </section>

      <section className="viewer-row main-viewers-row">
        <ImageViewer
          title="Fixed / Image + Segmentation"
          file={fixedFile}
          overlayFile={fixedSegFile}
          sliceIndex={sliceIndex}
          onSliceChange={handleSliceChange}
          mode="image"
          overlayEnabled={showOverlay}
          overlayOpacity={overlayOpacity}
          onPreviewInfoChange={(info) => setFixedIs3D(Boolean(info?.is3D))}
        />
        <ImageViewer
          title="Moving / Image + Segmentation"
          file={movingFile}
          overlayFile={movingSegFile}
          sliceIndex={sliceIndex}
          onSliceChange={handleSliceChange}
          mode="image"
          overlayEnabled={showOverlay}
          overlayOpacity={overlayOpacity}
          onPreviewInfoChange={(info) => setMovingIs3D(Boolean(info?.is3D))}
        />
        {fixedFile && movingFile && (
          <DifferenceViewer
            title="Difference Image"
            fixedFile={fixedFile}
            movingFile={movingFile}
            fixedSegFile={fixedSegFile}
            movingSegFile={movingSegFile}
            sliceIndex={sliceIndex}
          />
        )}
      </section>

      <section className="viewer-row secondary-viewers-row">
        <ImageViewer title="Displacement Field" file={dispFile} sliceIndex={sliceIndex} onSliceChange={handleSliceChange} mode="displacement" />
        {jacobianPreview && <JacobianViewer title="Jacobian Determinant" jacobian={jacobianPreview} sliceIndex={globalSliceIndex} onSliceChange={handleSliceChange} />}
      </section>

      <div className={`floating-actions ${actionMenuOpen ? "floating-actions-open" : ""}`}>
        <div className="floating-actions-menu" aria-hidden={!actionMenuOpen}>
          <button onClick={handleComputeMetrics} disabled={!canComputeMetrics}>
            Compute metrics
          </button>
          <button onClick={handleComputeJacobian} disabled={!canComputeJacobian}>
            Compute jacobian
          </button>
          <button className="secondary-action" onClick={handleGenerateSamples}>
            Generate samples
          </button>
        </div>
        <button className="floating-actions-toggle" onClick={() => setActionMenuOpen((open) => !open)}>
          {actionMenuOpen ? "Close" : "Actions"}
        </button>
      </div>
    </div>
  );
}

export default App;
