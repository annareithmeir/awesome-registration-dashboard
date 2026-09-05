import { getSegmentationColorCss } from "../utils/segmentationColors";

function StatRow({ label, value, wide }) {
  return (
    <div className={`stat-row ${wide ? "stat-row-wide" : ""}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

// Min/mean/max read as a range at a glance when they sit left-to-right in
// that order - min pinned left, max pinned right, mean centered between
// them - rather than as same-weight pills that wrap independently.
function MinMeanMaxRow({ min, mean, max, decimals = 4 }) {
  return (
    <div className="mmm-row">
      <div className="mmm-cell mmm-min">
        <span className="stat-label">Min</span>
        <span className="stat-value">{min.toFixed(decimals)}</span>
      </div>
      <div className="mmm-cell mmm-mean">
        <span className="stat-label">Mean</span>
        <span className="stat-value">{mean.toFixed(decimals)}</span>
      </div>
      <div className="mmm-cell mmm-max">
        <span className="stat-label">Max</span>
        <span className="stat-value">{max.toFixed(decimals)}</span>
      </div>
    </div>
  );
}

// Fixed per-metric hues (not per-structure) so Dice and HD read as two
// different kinds of measurement at a glance, no matter which label's row
// they're in - structure identity is already carried by the swatch/label
// text to the left of each group. Same family (teal), hue nudged toward
// blue for HD - a visible but not jarring difference, rather than two
// unrelated colors.
const DICE_COLOR = "#4fd6c5";
const HD_COLOR = "#4e94da";

// Compact in-row bar: fill length encodes this value on a shared [min, max]
// scale across all structures. The tick marks where the mean across all
// structures falls, so a glance at any row shows whether it's above or
// below the group average.
function MetricBar({ value, mean, min, max, color }) {
  const range = max - min || 1;
  const toPct = (v) => Math.max(0, Math.min(100, ((v - min) / range) * 100));
  const valuePct = toPct(value);
  const meanPct = toPct(mean);

  return (
    <span className="metric-bar" title={`${value.toFixed(4)} (mean ${mean.toFixed(4)})`}>
      <span className="metric-bar-track">
        <span className="metric-bar-fill" style={{ width: `${valuePct}%`, background: color }} />
        <span className="metric-bar-mean" style={{ left: `${meanPct}%` }} />
      </span>
    </span>
  );
}

function mean(values) {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

function MetricGroup({ className, title, children }) {
  return (
    <div className={`metric-group ${className}`}>
      <div className="metric-group-header">
        <span className="metric-group-dot" />
        <h3>{title}</h3>
      </div>
      {children}
    </div>
  );
}

function MetricPanel({ metrics, metricsBefore, jacobian }) {
  const showBefore = Boolean(metrics && metricsBefore);

  return (
    <div className="panel metrics-panel">
      <h2>Metrics</h2>

      <div className="metric-groups">
        <MetricGroup className="metric-group-segmentation" title="Segmentation Overlap">
          {metrics ? (
            <>
              <div className="stat-grid">
                {showBefore && <StatRow label="DSC₀" value={metricsBefore.dice?.toFixed(4)} />}
                <StatRow label="DSC" value={metrics.dice?.toFixed(4)} />
                <StatRow label="DSC95" value={metrics.dice95?.toFixed(4)} />
              </div>
              <div className="stat-grid">
                {showBefore && <StatRow label="HD₀" value={metricsBefore.hausdorff?.toFixed(4)} />}
                <StatRow label="HD" value={metrics.hausdorff?.toFixed(4)} />
                <StatRow label="HD95" value={metrics.hausdorff95?.toFixed(4)} />
              </div>
              {Array.isArray(metrics.per_structure) && metrics.per_structure.length > 0 && (() => {
                const structures = metrics.per_structure;
                const diceValues = structures.map((s) => s.dice).filter(Number.isFinite);
                const hdValues = structures.map((s) => s.hausdorff).filter(Number.isFinite);
                const diceMean = mean(diceValues);
                const hdMean = mean(hdValues);

                const beforeByLabel = new Map((metricsBefore?.per_structure || []).map((s) => [s.label, s]));
                const beforeDiceValues = structures
                  .map((s) => beforeByLabel.get(s.label)?.dice)
                  .filter(Number.isFinite);
                const beforeHdValues = structures
                  .map((s) => beforeByLabel.get(s.label)?.hausdorff)
                  .filter(Number.isFinite);
                const beforeDiceMean = mean(beforeDiceValues);
                const beforeHdMean = mean(beforeHdValues);

                // Before/after HD bars share one scale (the union of both sets)
                // so bar length is directly comparable between the two rows,
                // not just within each row on its own.
                const allHdValues = [...hdValues, ...beforeHdValues];
                const hdMin = allHdValues.length ? Math.min(...allHdValues) : 0;
                const hdMax = allHdValues.length ? Math.max(...allHdValues) : 1;

                return (
                  <div className="structure-metrics">
                    <div className="structure-metrics-title">Per-structure ({structures.length})</div>
                    <div className="metric-group-subtitle">bar = value, tick = mean across labels</div>
                    <div className="structure-metrics-list">
                      {structures.map((item) => {
                        const color = getSegmentationColorCss(item.label);
                        const beforeItem = beforeByLabel.get(item.label);
                        return (
                          <div className="structure-metric-row" key={item.label}>
                            <div className="structure-label">
                              <span className="legend-swatch structure-swatch" style={{ background: color }} />
                              <span>Label {item.label}</span>
                            </div>
                            <div className="structure-metric-lines">
                              {showBefore && Number.isFinite(beforeItem?.dice) && (
                                <div className="structure-metric-line structure-metric-line-before">
                                  <span className="structure-metric-name">Dice₀</span>
                                  <MetricBar value={beforeItem.dice} mean={beforeDiceMean} min={0} max={1} color={DICE_COLOR} />
                                  <span className="stat-value">{beforeItem.dice.toFixed(4)}</span>
                                </div>
                              )}
                              <div className="structure-metric-line">
                                <span className="structure-metric-name">Dice</span>
                                <MetricBar value={item.dice} mean={diceMean} min={0} max={1} color={DICE_COLOR} />
                                <span className="stat-value">{item.dice?.toFixed(4)}</span>
                              </div>
                              {showBefore && Number.isFinite(beforeItem?.hausdorff) && (
                                <div className="structure-metric-line structure-metric-line-before">
                                  <span className="structure-metric-name">HD₀</span>
                                  <MetricBar value={beforeItem.hausdorff} mean={beforeHdMean} min={hdMin} max={hdMax} color={HD_COLOR} />
                                  <span className="stat-value">{beforeItem.hausdorff.toFixed(4)}</span>
                                </div>
                              )}
                              <div className="structure-metric-line">
                                <span className="structure-metric-name">HD</span>
                                <MetricBar value={item.hausdorff} mean={hdMean} min={hdMin} max={hdMax} color={HD_COLOR} />
                                <span className="stat-value">{item.hausdorff?.toFixed(4)}</span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })()}
            </>
          ) : (
            <div className="empty-state">Upload fixed + moving segmentation files to compute metrics.</div>
          )}
        </MetricGroup>

        {jacobian && (
          <>
            <div className="metric-column">
              <MetricGroup className="metric-group-jacobian" title="Jacobian Determinant">
                <div className={`topology-badge ${jacobian.frac_negative <= 0 ? "topology-ok" : "topology-warning"}`}>
                  {jacobian.frac_negative <= 0
                    ? "Topology preserving — no folding"
                    : `Folding — ${(jacobian.frac_negative * 100).toFixed(2)}% of voxels`}
                </div>
                <MinMeanMaxRow min={jacobian.min} mean={jacobian.mean} max={jacobian.max} />
                <div className="stat-grid">
                  <StatRow label="Frac. negative" value={jacobian.frac_negative.toFixed(4)} wide />
                </div>
              </MetricGroup>

              {jacobian.log_jacobian && (
                <MetricGroup className="metric-group-logjac" title="Log-Jacobian">
                  <div className="metric-group-subtitle">ln J, J&gt;0 — symmetric growth/shrinkage measure</div>
                  <div className="stat-grid">
                    <StatRow label="Mean" value={jacobian.log_jacobian.mean.toFixed(4)} />
                    <StatRow label="Std" value={jacobian.log_jacobian.std.toFixed(4)} />
                    {jacobian.log_jacobian.excluded_fraction > 0 && (
                      <StatRow
                        label="Excluded (folded)"
                        value={`${(jacobian.log_jacobian.excluded_fraction * 100).toFixed(2)}%`}
                        wide
                      />
                    )}
                  </div>
                </MetricGroup>
              )}
            </div>

            <div className="metric-column">
              {jacobian.shear && (
                <MetricGroup className="metric-group-shear" title="Shear Index">
                  <div className="metric-group-subtitle">max/min singular value of local Jacobian</div>
                  <MinMeanMaxRow min={jacobian.shear.min} mean={jacobian.shear.mean} max={jacobian.shear.max} />
                </MetricGroup>
              )}

              {jacobian.inverse_consistency && (
                <MetricGroup className="metric-group-ice" title="Inverse Consistency Error">
                  <MinMeanMaxRow
                    min={jacobian.inverse_consistency.min}
                    mean={jacobian.inverse_consistency.mean}
                    max={jacobian.inverse_consistency.max}
                  />
                </MetricGroup>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default MetricPanel;
