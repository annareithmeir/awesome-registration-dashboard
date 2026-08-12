function MetricPanel({ metrics, jacobian }) {
  return (
    <div className="panel metrics-panel">
      <h2>Metrics</h2>
      {metrics ? (
        <div className="metrics-block">
          <div className="metric-section">
            <div className="metric-section-label">Mean</div>
            <div className="metrics-inline">
              <div className="metric-pill">Dice {metrics.dice?.toFixed(4)}</div>
              <div className="metric-pill">Dice95 {metrics.dice95?.toFixed(4)}</div>
              <div className="metric-pill">Hausdorff {metrics.hausdorff?.toFixed(4)}</div>
              <div className="metric-pill">Hausdorff95 {metrics.hausdorff95?.toFixed(4)}</div>
            </div>
          </div>
          {Array.isArray(metrics.per_structure) && metrics.per_structure.length > 0 && (
            <div className="structure-metrics">
              <div className="structure-metrics-title">Per-structure</div>
              <div className="structure-metrics-list">
                {metrics.per_structure.map((item) => (
                  <div className="structure-metric-row" key={item.label}>
                    <div className="structure-label">
                      <span className={`legend-swatch structure-swatch structure-swatch-${item.label}`} />
                      <span>Label {item.label}</span>
                    </div>
                    <div className="structure-values">
                      <span className="metric-pill">Dice {item.dice?.toFixed(4)}</span>
                      <span className="metric-pill">Hausdorff {item.hausdorff?.toFixed(4)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="empty-state">Upload fixed + moving segmentation files to compute metrics.</div>
      )}

      <h2>Jacobian</h2>
      {jacobian ? (
        <div className="metrics-inline">
          <div className="metric-pill">Min {jacobian.min.toFixed(4)}</div>
          <div className="metric-pill">Max {jacobian.max.toFixed(4)}</div>
          <div className="metric-pill">Mean {jacobian.mean.toFixed(4)}</div>
          <div className="metric-pill">Frac negative {jacobian.frac_negative.toFixed(4)}</div>
        </div>
      ) : (
        <div className="empty-state">Upload a displacement `.pt` file to compute jacobian.</div>
      )}
    </div>
  );
}

export default MetricPanel;
