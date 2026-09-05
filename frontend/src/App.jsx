import { useEffect, useRef, useState } from "react";
import ImageViewer from "./components/ImageViewer";
import DifferenceViewer from "./components/DifferenceViewer";
import MetricPanel from "./components/MetricPanel";
import JacobianViewer from "./components/JacobianViewer";
import ScalarFieldViewer from "./components/ScalarFieldViewer";
import SampleThumbnail from "./components/SampleThumbnail";
import "./styles.css";

const SAMPLE_SETS = [
  {
    id: "expand-shrink",
    label: "Expand / Shrink",
    description: "Synthetic circles — one expanding, one shrinking",
    fixedName: "img2d_fixed.nii.gz",
    movingName: "img2d_moving.nii.gz",
    fixedSegName: "seg2d_fixed.nii.gz",
    movingSegName: "seg2d_moving.nii.gz",
  },
  {
    id: "acdc",
    label: "ACDC Patient 19",
    description: "Cardiac MRI, cropped around the heart — ED vs. ES phase",
    fixedName: "acdc_fixed.nii.gz",
    movingName: "acdc_moving.nii.gz",
    fixedSegName: "acdc_fixed_seg.nii.gz",
    movingSegName: "acdc_moving_seg.nii.gz",
  },
  {
    id: "nlst",
    label: "NLST Patient 1 (3D)",
    description: "Lung CT, cropped + downsampled — baseline vs. follow-up",
    fixedName: "nlst_fixed.nii.gz",
    movingName: "nlst_moving.nii.gz",
    fixedSegName: "nlst_fixed_seg.nii.gz",
    movingSegName: "nlst_moving_seg.nii.gz",
  },
];

// Volumes loaded here (both the bundled samples and typical NIfTI exports)
// follow the standard radiological axis order - array axis 0 is the
// left-right (sagittal) axis, axis 1 is anterior-posterior (coronal), axis 2
// is inferior-superior (axial) - so the slice-axis picker can label them
// anatomically instead of just "axis 0/1/2".
const AXIS_OPTIONS = [
  { value: 0, label: "Sagittal" },
  { value: 1, label: "Coronal" },
  { value: 2, label: "Axial" },
];

function App() {
  const [fixedFile, setFixedFile] = useState(null);
  const [movingFile, setMovingFile] = useState(null);
  const [fixedSegFile, setFixedSegFile] = useState(null);
  const [movingSegFile, setMovingSegFile] = useState(null);
  const [dispFile, setDispFile] = useState(null);
  const [warpedFile, setWarpedFile] = useState(null);
  const [warpedSegFile, setWarpedSegFile] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [metricsBefore, setMetricsBefore] = useState(null);
  const [jacobian, setJacobian] = useState(null);
  const [jacobianPreview, setJacobianPreview] = useState(null);
  const [sliceIndex, setSliceIndex] = useState(0);
  const [globalSliceIndex, setGlobalSliceIndex] = useState(0);
  const [showOverlay, setShowOverlay] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(0.65);
  const [fixedIs3D, setFixedIs3D] = useState(false);
  const [movingIs3D, setMovingIs3D] = useState(false);
  const [fixedSliceCount, setFixedSliceCount] = useState(1);
  const [movingSliceCount, setMovingSliceCount] = useState(1);
  const [sliceAxis, setSliceAxis] = useState(0);
  const [rotation, setRotation] = useState(0);
  const [actionMenuOpen, setActionMenuOpen] = useState(false);
  const [dataMenuOpen, setDataMenuOpen] = useState(true);
  const [showSamplePicker, setShowSamplePicker] = useState(false);
  const [sampleUrls, setSampleUrls] = useState({});
  const [loadingSampleSetId, setLoadingSampleSetId] = useState(null);
  const [showRegisteredLoaders, setShowRegisteredLoaders] = useState(false);
  const [showRegistrationOptions, setShowRegistrationOptions] = useState(false);
  const [registrationType, setRegistrationType] = useState("deformable");
  const [bendingEnergy, setBendingEnergy] = useState(0.005);
  const [maxIterations, setMaxIterations] = useState(750);
  const [gridSpacing, setGridSpacing] = useState(5);
  const [registrationRunning, setRegistrationRunning] = useState(false);
  const [registrationPhase, setRegistrationPhase] = useState(null);
  const [registrationProgress, setRegistrationProgress] = useState(0);
  const [registrationError, setRegistrationError] = useState(null);
  const canUseSliceSlider = fixedIs3D && movingIs3D;
  const maxSliceIndex = Math.max(0, Math.min(fixedSliceCount, movingSliceCount) - 1);
  const canRunRegistration = Boolean(fixedFile && movingFile) && !registrationRunning;

  const handleSliceChange = (value) => {
    setSliceIndex(value);
    setGlobalSliceIndex(value);
  };

  // Switching axis changes which dimension the slice index runs along, so
  // whatever position was picked on the old axis has no meaning on the new
  // one - reset back to the first slice rather than clamping into range.
  const handleAxisChange = (value) => {
    setSliceAxis(value);
    handleSliceChange(0);
  };

  // Purely a display transform - rotates every viewer's current slice 90°
  // clockwise per click, in lockstep. Doesn't touch the underlying volume
  // data or any computed value (metrics, Jacobian, ...), only how it's drawn.
  const handleRotate = () => {
    setRotation((current) => (current + 90) % 360);
  };

  // Every slice change re-fetches previews/metrics/jacobian for every open
  // viewer, so committing on literally every step while dragging the slider
  // floods the backend and the UI visibly lags behind the mouse. Track the
  // slider's own live position separately, and only commit (i.e. actually
  // trigger those fetches) every few slices while dragging - still gives
  // feedback as you drag, without a request per single slice. Releasing the
  // slider (or using the keyboard) always commits the exact final value.
  const [sliceDisplayValue, setSliceDisplayValue] = useState(0);
  useEffect(() => {
    setSliceDisplayValue(globalSliceIndex);
  }, [globalSliceIndex]);
  // Each commit fires a preview/metrics/jacobian fetch for every open tile at
  // once, so keep the number of commits over a full drag modest (~15) rather
  // than merely halving it - the lag comes from request pileup, not from the
  // per-request cost alone.
  const sliceDragStep = Math.max(1, Math.round(maxSliceIndex / 15));

  const handleSliceDrag = (value) => {
    setSliceDisplayValue(value);
    if (value % sliceDragStep === 0 || value === maxSliceIndex) {
      handleSliceChange(value);
    }
  };

  const handleSliceCommit = (value) => {
    setSliceDisplayValue(value);
    handleSliceChange(value);
  };

  // Mouse-wheel slice scrubbing, active only while the cursor is actually
  // over a plot (a .viewer-preview box) - not the header, controls, gaps
  // between tiles, or the rest of the page, which all keep scrolling the
  // page normally. Same underlying sliceIndex as the slider, so both stay
  // in sync and either can be used at any time. A standard mouse wheel
  // notch reports deltaY ~100, so that's the threshold for "one slice
  // step"; a trackpad's much smaller deltas accumulate toward it instead of
  // skipping several slices per event. Committing (the actual
  // fetch-triggering update) is debounced the same way slider drags are - a
  // fast flick shows intermediate slices instantly but only the slice you
  // stop on fires the backend requests.
  //
  // This has to be a native (non-React) listener: React attaches `onWheel`
  // as a passive DOM listener by default, so e.preventDefault() inside a
  // React handler is silently ignored and the page would scroll along with
  // the slice change instead of staying put. Listening on the document
  // (rather than each tile) and checking the event's target means there's
  // nothing to enable/disable on click - it's simply live wherever the
  // cursor happens to be.
  const wheelCommitTimeoutRef = useRef(null);

  useEffect(() => {
    const WHEEL_STEP_DELTA = 100;
    const wheelAccum = { current: 0 };

    const handleWheel = (e) => {
      if (!canUseSliceSlider) return;
      if (!e.target.closest(".viewer-preview")) return;
      e.preventDefault();
      wheelAccum.current += e.deltaY;
      if (Math.abs(wheelAccum.current) < WHEEL_STEP_DELTA) return;
      const steps = Math.trunc(wheelAccum.current / WHEEL_STEP_DELTA);
      wheelAccum.current -= steps * WHEEL_STEP_DELTA;

      setSliceDisplayValue((current) => {
        const next = Math.max(0, Math.min(maxSliceIndex, current + steps));
        if (wheelCommitTimeoutRef.current) clearTimeout(wheelCommitTimeoutRef.current);
        wheelCommitTimeoutRef.current = setTimeout(() => handleSliceChange(next), 120);
        return next;
      });
    };

    document.addEventListener("wheel", handleWheel, { passive: false });
    return () => document.removeEventListener("wheel", handleWheel);
  }, [canUseSliceSlider, maxSliceIndex]);

  const computeMetrics = async (fixedSeg, movingSeg) => {
    if (!fixedSeg || !movingSeg) return null;
    const formData = new FormData();
    formData.append("fixed_seg", fixedSeg);
    formData.append("moving_seg", movingSeg);
    const response = await fetch("http://localhost:8000/compute-metrics", {
      method: "POST",
      body: formData,
    });
    return response.json();
  };

  const computeJacobian = async (disp, atSliceIndex, atAxis) => {
    if (!disp) return;
    const formData = new FormData();
    formData.append("disp_field", disp);
    formData.append("slice_index", String(atSliceIndex));
    formData.append("axis", String(atAxis));
    const response = await fetch("http://localhost:8000/compute-jacobian", {
      method: "POST",
      body: formData,
    });
    const json = await response.json();
    setJacobian(json);
    setJacobianPreview(json);
  };

  // Metrics and jacobian are always kept up to date automatically, rather
  // than requiring a manual "compute" click. Metrics compare the fixed
  // segmentation against whatever the best available match is (the warped
  // segmentation once registration has produced one, else the moving
  // segmentation) - and separately, "before" metrics always compare fixed
  // vs. moving directly, so the improvement from registration is visible
  // once a warped segmentation exists. The jacobian recomputes whenever the
  // displacement field or the active slice changes.
  useEffect(() => {
    const fixedSeg = fixedSegFile || fixedFile;
    const compareSeg = warpedSegFile || movingSegFile || movingFile;
    let canceled = false;
    if (fixedSeg && compareSeg) {
      computeMetrics(fixedSeg, compareSeg).then((result) => {
        if (!canceled) setMetrics(result);
      });
    } else {
      setMetrics(null);
    }
    return () => {
      canceled = true;
    };
  }, [fixedSegFile, fixedFile, warpedSegFile, movingSegFile, movingFile]);

  useEffect(() => {
    const fixedSeg = fixedSegFile || fixedFile;
    const movingSeg = movingSegFile || movingFile;
    let canceled = false;
    if (warpedSegFile && fixedSeg && movingSeg) {
      computeMetrics(fixedSeg, movingSeg).then((result) => {
        if (!canceled) setMetricsBefore(result);
      });
    } else {
      setMetricsBefore(null);
    }
    return () => {
      canceled = true;
    };
  }, [warpedSegFile, fixedSegFile, fixedFile, movingSegFile, movingFile]);

  useEffect(() => {
    if (dispFile) {
      computeJacobian(dispFile, globalSliceIndex, sliceAxis);
    } else {
      setJacobian(null);
      setJacobianPreview(null);
    }
  }, [dispFile, globalSliceIndex, sliceAxis]);

  // Once the registered-data loader panel has produced both a warped image
  // and a displacement field, it has done its job — collapse it.
  useEffect(() => {
    if (warpedFile && dispFile) {
      setShowRegisteredLoaders(false);
    }
  }, [warpedFile, dispFile]);

  // reg_aladin and reg_f3d have very different natural iteration counts
  // (NiftyReg defaults: 5 vs. 150), so reset to a sensible starting point
  // whenever the registration mode changes; still fully overridable below.
  useEffect(() => {
    setMaxIterations(registrationType === "affine" ? 25 : 750);
  }, [registrationType]);

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

  const pollRegistrationStatus = (jobId) =>
    new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const response = await fetch(`http://localhost:8000/registration-status/${jobId}`);
          if (!response.ok) throw new Error("Unable to fetch registration status");
          const json = await response.json();
          setRegistrationPhase(json.phase);
          if (typeof json.progress === "number") setRegistrationProgress(json.progress);

          if (json.status === "done") {
            resolve(json.results);
          } else if (json.status === "error") {
            reject(new Error(json.error || "Registration failed"));
          } else {
            setTimeout(poll, 700);
          }
        } catch (error) {
          reject(error);
        }
      };
      poll();
    });

  const handleStartRegistration = async () => {
    if (!canRunRegistration) return;
    setShowRegistrationOptions(false);
    setRegistrationRunning(true);
    setRegistrationError(null);
    setRegistrationProgress(0);
    setRegistrationPhase("Starting…");
    try {
      const formData = new FormData();
      formData.append("fixed", fixedFile);
      formData.append("moving", movingFile);
      if (movingSegFile) formData.append("moving_seg", movingSegFile);
      formData.append("registration_type", registrationType);
      formData.append("max_iterations", String(maxIterations));
      if (registrationType === "deformable") {
        formData.append("bending_energy", String(bendingEnergy));
        formData.append("grid_spacing", String(gridSpacing));
      }

      const response = await fetch("http://localhost:8000/run-registration", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Unable to start registration");
      const { job_id: jobId } = await response.json();

      const results = await pollRegistrationStatus(jobId);
      setRegistrationPhase("Done");
      setRegistrationProgress(1);

      const [warpedFileObj, warpedSegFileObj, dispFileObj] = await Promise.all([
        loadSampleFile(results.warped_image),
        results.warped_segmentation ? loadSampleFile(results.warped_segmentation) : Promise.resolve(null),
        loadSampleFile(results.displacement),
      ]);
      setWarpedFile(warpedFileObj);
      setWarpedSegFile(warpedSegFileObj);
      setDispFile(dispFileObj);
      setActionMenuOpen(false);
    } catch (error) {
      console.error("Registration failed", error);
      setRegistrationError(error.message || "Registration failed");
    } finally {
      setRegistrationRunning(false);
      setTimeout(() => setRegistrationPhase(null), 1500);
    }
  };

  const handleOpenSamplePicker = async () => {
    setShowSamplePicker((open) => !open);
    if (Object.keys(sampleUrls).length > 0) return;
    try {
      const response = await fetch("http://localhost:8000/generate-samples", { method: "POST" });
      const json = await response.json();
      const map = {};
      (json.urls || []).forEach((url) => {
        map[url.split("/").pop()] = url;
      });
      setSampleUrls(map);
    } catch (error) {
      console.error("Unable to load sample list", error);
    }
  };

  const handleSelectSampleSet = async (set) => {
    if (loadingSampleSetId) return;
    const fixedUrl = sampleUrls[set.fixedName];
    const movingUrl = sampleUrls[set.movingName];
    if (!fixedUrl || !movingUrl) return;
    const fixedSegUrl = sampleUrls[set.fixedSegName];
    const movingSegUrl = sampleUrls[set.movingSegName];
    setLoadingSampleSetId(set.id);
    try {
      await Promise.all([
        handleSetSample(fixedUrl, setFixedFile),
        handleSetSample(movingUrl, setMovingFile),
        fixedSegUrl ? handleSetSample(fixedSegUrl, setFixedSegFile) : Promise.resolve(),
        movingSegUrl ? handleSetSample(movingSegUrl, setMovingSegFile) : Promise.resolve(),
      ]);
    } finally {
      setLoadingSampleSetId(null);
      setShowSamplePicker(false);
      setDataMenuOpen(false);
    }
  };

  return (
    <div className="app-shell">
      <header>
        <h1>Awesome Image Registration Dashboard</h1>
        <p>Load two scans, register them, and see exactly how well it worked — down to the last voxel.</p>
      </header>

      <section className="metrics-row">
        <MetricPanel metrics={metrics} metricsBefore={metricsBefore} jacobian={jacobian} />
      </section>

      <section className="viewer-row main-viewers-row">
        <ImageViewer
          title="Fixed"
          file={fixedFile}
          overlayFile={fixedSegFile}
          sliceIndex={sliceIndex}
          onSliceChange={handleSliceChange}
          axis={sliceAxis}
          rotation={rotation}
          mode="image"
          overlayEnabled={showOverlay}
          overlayOpacity={overlayOpacity}
          onOverlayEnabledChange={setShowOverlay}
          onPreviewInfoChange={(info) => {
            setFixedIs3D(Boolean(info?.is3D));
            setFixedSliceCount(info?.sliceCount || 1);
          }}
        />
        <ImageViewer
          title="Moving"
          file={movingFile}
          overlayFile={movingSegFile}
          sliceIndex={sliceIndex}
          onSliceChange={handleSliceChange}
          axis={sliceAxis}
          rotation={rotation}
          mode="image"
          overlayEnabled={showOverlay}
          overlayOpacity={overlayOpacity}
          onOverlayEnabledChange={setShowOverlay}
          onPreviewInfoChange={(info) => {
            setMovingIs3D(Boolean(info?.is3D));
            setMovingSliceCount(info?.sliceCount || 1);
          }}
        />
        {fixedFile && movingFile && (
          <DifferenceViewer
            title="Difference Image"
            fixedFile={fixedFile}
            movingFile={movingFile}
            fixedSegFile={fixedSegFile}
            movingSegFile={movingSegFile}
            warpedFile={warpedFile}
            warpedSegFile={warpedSegFile}
            sliceIndex={sliceIndex}
            axis={sliceAxis}
            rotation={rotation}
            showSegmentations={showOverlay}
          />
        )}
      </section>

      {(warpedFile || dispFile) && (
        <section className="viewer-row secondary-viewers-row">
          <ImageViewer
            title="Warped"
            file={warpedFile}
            overlayFile={warpedSegFile}
            sliceIndex={sliceIndex}
            onSliceChange={handleSliceChange}
            axis={sliceAxis}
            rotation={rotation}
            mode="image"
            overlayEnabled={showOverlay}
            overlayOpacity={overlayOpacity}
            onOverlayEnabledChange={setShowOverlay}
          />
          <ImageViewer
            title="Displacement Field"
            file={dispFile}
            backgroundFile={fixedFile}
            sliceIndex={sliceIndex}
            onSliceChange={handleSliceChange}
            axis={sliceAxis}
            rotation={rotation}
            mode="displacement"
          />
          {jacobianPreview && (
            <JacobianViewer title="Jacobian Determinant" jacobian={jacobianPreview} rotation={rotation} />
          )}
          {jacobianPreview?.shear && (
            <ScalarFieldViewer
              title="Shear Index"
              field={jacobianPreview.shear}
              rotation={rotation}
              lowColor={[45, 156, 219]}
              highColor={[240, 164, 66]}
            />
          )}
          {jacobianPreview?.inverse_consistency && (
            <ScalarFieldViewer
              title="Inverse Consistency Error"
              field={jacobianPreview.inverse_consistency}
              rotation={rotation}
              lowColor={[45, 156, 219]}
              highColor={[222, 82, 75]}
            />
          )}
        </section>
      )}

      <div className="floating-corner">
        {showOverlay && (
          <div className="floating-overlay-control">
            <label>
              Overlay opacity: {Math.round(overlayOpacity * 100)}%
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(Number(e.target.value))}
              />
            </label>
          </div>
        )}
        <div className={`floating-menu-group ${dataMenuOpen ? "floating-menu-open" : ""}`}>
          <div className="floating-menu-panel data-menu-panel" aria-hidden={!dataMenuOpen}>
            <button className="secondary-action data-menu-span" onClick={handleOpenSamplePicker}>
              {showSamplePicker ? "Hide sample data" : "Load sample data"}
            </button>
            {showSamplePicker && (
              <div className="sample-picker data-menu-span">
                {SAMPLE_SETS.map((set) => {
                  const fixedUrl = sampleUrls[set.fixedName];
                  const movingUrl = sampleUrls[set.movingName];
                  const isLoading = loadingSampleSetId === set.id;
                  return (
                    <button
                      key={set.id}
                      className="sample-set-option"
                      onClick={() => handleSelectSampleSet(set)}
                      disabled={!fixedUrl || !movingUrl || (Boolean(loadingSampleSetId) && !isLoading)}
                    >
                      <div className="sample-set-label">
                        {set.label}
                        {isLoading && <span className="viewer-spinner sample-set-spinner" />}
                      </div>
                      <div className="sample-set-description">{isLoading ? "Loading…" : set.description}</div>
                      <div className="sample-set-thumbs">
                        {fixedUrl && <SampleThumbnail url={fixedUrl} label="Fixed" />}
                        {movingUrl && <SampleThumbnail url={movingUrl} label="Moving" />}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            <label className="file-card">
              Fixed image
              <input type="file" onChange={(e) => setFixedFile(e.target.files?.[0] ?? null)} />
            </label>
            <label className="file-card">
              Moving image
              <input type="file" onChange={(e) => setMovingFile(e.target.files?.[0] ?? null)} />
            </label>
            <label className="file-card">
              Fixed segmentation
              <input type="file" onChange={(e) => setFixedSegFile(e.target.files?.[0] ?? null)} />
            </label>
            <label className="file-card">
              Moving segmentation
              <input type="file" onChange={(e) => setMovingSegFile(e.target.files?.[0] ?? null)} />
            </label>

            <button
              className="secondary-action data-menu-span"
              onClick={() => setShowRegisteredLoaders((open) => !open)}
            >
              {showRegisteredLoaders ? "Hide registered data" : "Load registered data"}
            </button>
            {showRegisteredLoaders && (
              <>
                <label className="file-card">
                  Warped image
                  <input type="file" onChange={(e) => setWarpedFile(e.target.files?.[0] ?? null)} />
                </label>
                <label className="file-card">
                  Warped segmentation
                  <input type="file" onChange={(e) => setWarpedSegFile(e.target.files?.[0] ?? null)} />
                </label>
                <label className="file-card data-menu-span">
                  Displacement field (.pt)
                  <input type="file" onChange={(e) => setDispFile(e.target.files?.[0] ?? null)} />
                </label>
              </>
            )}
          </div>
          <button className="floating-menu-toggle" onClick={() => setDataMenuOpen((open) => !open)}>
            {dataMenuOpen ? "Close" : "Data"}
          </button>
        </div>

        <div className={`floating-menu-group ${actionMenuOpen ? "floating-menu-open" : ""}`}>
          <div className="floating-menu-panel" aria-hidden={!actionMenuOpen}>
            <button onClick={() => setShowRegistrationOptions((open) => !open)} disabled={!canRunRegistration}>
              {registrationRunning ? "Running registration…" : "Run registration"}
            </button>

            {showRegistrationOptions && (
              <div className="registration-inline-settings">
                <div className="registration-popover-title">Registration settings</div>
                <div className="registration-type-options">
                  <label className="radio-control">
                    <input
                      type="radio"
                      name="registration-type"
                      value="deformable"
                      checked={registrationType === "deformable"}
                      onChange={() => setRegistrationType("deformable")}
                    />
                    Deformable (reg_f3d)
                  </label>
                  <label className="radio-control">
                    <input
                      type="radio"
                      name="registration-type"
                      value="affine"
                      checked={registrationType === "affine"}
                      onChange={() => setRegistrationType("affine")}
                    />
                    Affine (reg_aladin)
                  </label>
                </div>
                <label className="popover-slider">
                  Max iterations (per level): {maxIterations}
                  <input
                    type="range"
                    min={registrationType === "affine" ? 5 : 50}
                    max={registrationType === "affine" ? 200 : 2000}
                    step={registrationType === "affine" ? 5 : 25}
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(Number(e.target.value))}
                  />
                </label>
                {registrationType === "deformable" && (
                  <>
                    <label className="popover-slider">
                      Regularization weight: {bendingEnergy.toFixed(3)}
                      <input
                        type="range"
                        min="0"
                        max="0.05"
                        step="0.001"
                        value={bendingEnergy}
                        onChange={(e) => setBendingEnergy(Number(e.target.value))}
                      />
                    </label>
                    <label className="popover-slider">
                      Control point spacing: {gridSpacing.toFixed(1)} voxels
                      <input
                        type="range"
                        min="1"
                        max="10"
                        step="0.5"
                        value={gridSpacing}
                        onChange={(e) => setGridSpacing(Number(e.target.value))}
                      />
                    </label>
                  </>
                )}
                <div className="registration-popover-actions">
                  <button onClick={handleStartRegistration} disabled={!canRunRegistration}>
                    Start registration
                  </button>
                  <button className="secondary-action" onClick={() => setShowRegistrationOptions(false)}>
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {registrationPhase && (
              <div className="registration-progress">
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${Math.round(registrationProgress * 100)}%` }} />
                </div>
                <div className="progress-label">
                  {registrationPhase} {registrationRunning && `(${Math.round(registrationProgress * 100)}%)`}
                </div>
              </div>
            )}
            {registrationError && <div className="sample-message error-message">{registrationError}</div>}
          </div>
          <button className="floating-menu-toggle" onClick={() => setActionMenuOpen((open) => !open)}>
            {actionMenuOpen ? "Close" : "Actions"}
          </button>
        </div>
      </div>

      {canUseSliceSlider && (
        <div className="floating-slice-control">
          <div className="axis-control">
            {AXIS_OPTIONS.map(({ value, label }) => (
              <button
                key={value}
                type="button"
                className={value === sliceAxis ? "axis-option axis-option-active" : "axis-option"}
                onClick={() => handleAxisChange(value)}
                title={`${label} plane`}
              >
                {label}
              </button>
            ))}
          </div>
          <label>
            Slice: {Math.min(sliceDisplayValue, maxSliceIndex) + 1} / {maxSliceIndex + 1}
            <input
              type="range"
              min="0"
              max={maxSliceIndex}
              value={Math.min(sliceDisplayValue, maxSliceIndex)}
              onChange={(e) => handleSliceDrag(Number(e.target.value))}
              onMouseUp={(e) => handleSliceCommit(Number(e.target.value))}
              onTouchEnd={(e) => handleSliceCommit(Number(e.target.value))}
              onKeyUp={(e) => handleSliceCommit(Number(e.target.value))}
            />
          </label>
          <button
            type="button"
            className="icon-button rotate-button"
            onClick={handleRotate}
            title="Rotate view 90°"
            aria-label="Rotate view 90°"
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12a9 9 0 1 0 3-6.7" />
              <polyline points="3 4 3 8 7 8" />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
