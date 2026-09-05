// Shared per-label color scheme, used by both the segmentation overlay
// canvas (ImageViewer) and the per-structure metric swatches (MetricPanel)
// so a label reads as the same color everywhere. Labels 1 and 2 keep fixed,
// familiar colors; any further label gets a deterministic hue via
// golden-angle rotation, which spreads arbitrarily many labels across
// well-separated hues instead of collapsing them onto one fixed color.
function hslToRgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;

  let r = 0;
  let g = 0;
  let b = 0;

  if (h < 60) {
    r = c; g = x; b = 0;
  } else if (h < 120) {
    r = x; g = c; b = 0;
  } else if (h < 180) {
    r = 0; g = c; b = x;
  } else if (h < 240) {
    r = 0; g = x; b = c;
  } else if (h < 300) {
    r = x; g = 0; b = c;
  } else {
    r = c; g = 0; b = x;
  }

  return [r + m, g + m, b + m];
}

export function getSegmentationColor(value) {
  const numericValue = Number(value) || 0;
  if (!numericValue) {
    return [0, 0, 0];
  }

  if (numericValue === 1) {
    return [45, 156, 219];
  }

  if (numericValue === 2) {
    return [241, 146, 32];
  }

  const goldenAngle = 137.508;
  const hue = (300 + (numericValue - 3) * goldenAngle) % 360;
  const sat = 0.75;
  const light = 0.55;
  const [r, g, b] = hslToRgb(hue, sat, light);
  return [r * 255, g * 255, b * 255];
}

export function getSegmentationColorCss(value) {
  const [r, g, b] = getSegmentationColor(value);
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}
