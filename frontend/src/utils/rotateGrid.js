// Rotates a 2D row-major grid (array of arrays) clockwise by a multiple of
// 90 degrees, purely for display - it's applied after data is fetched, so it
// never touches the underlying volume or any computed value (metrics,
// Jacobian, ...), only how a slice is drawn. For 90/270 the returned grid's
// dimensions are swapped (rows become cols and vice versa), matching a real
// rotation of the image.
export function rotateGrid(data, rotation) {
  if (!data || !data.length || !rotation) return data;
  const rows = data.length;
  const cols = data[0]?.length || 0;
  if (!cols) return data;

  if (rotation === 180) {
    return data.map((row, r) => row.map((_, c) => data[rows - 1 - r][cols - 1 - c]));
  }

  const result = [];
  if (rotation === 90) {
    for (let i = 0; i < cols; i += 1) {
      const row = [];
      for (let j = 0; j < rows; j += 1) {
        row.push(data[rows - 1 - j][i]);
      }
      result.push(row);
    }
  } else if (rotation === 270) {
    for (let i = 0; i < cols; i += 1) {
      const row = [];
      for (let j = 0; j < rows; j += 1) {
        row.push(data[j][cols - 1 - i]);
      }
      result.push(row);
    }
  } else {
    return data;
  }
  return result;
}

// Companion to rotateGrid for the displacement-field warp grid: warpX/warpY
// store each point's warped pixel *position* (in the original, unrotated
// coordinate space), not just a value at that position - so on top of
// reindexing them like any other grid, the coordinates themselves need to be
// remapped into the rotated frame, via the same rotation applied to the
// (x, y) pair.
export function rotateWarpGrid(warpX, warpY, rotation) {
  if (!warpX || !warpY || !rotation) return { warpX, warpY };
  const rows = warpX.length;
  const cols = rows > 0 ? warpX[0].length : 0;
  if (!cols) return { warpX, warpY };

  const rotatedX = rotateGrid(warpX, rotation);
  const rotatedY = rotateGrid(warpY, rotation);

  const transform = (x, y) => {
    if (rotation === 90) return [rows - 1 - y, x];
    if (rotation === 180) return [cols - 1 - x, rows - 1 - y];
    if (rotation === 270) return [y, cols - 1 - x];
    return [x, y];
  };

  const outRows = rotatedX.length;
  const outCols = outRows > 0 ? rotatedX[0].length : 0;
  const outX = [];
  const outY = [];
  for (let i = 0; i < outRows; i += 1) {
    const rowX = [];
    const rowY = [];
    for (let j = 0; j < outCols; j += 1) {
      const [tx, ty] = transform(rotatedX[i][j], rotatedY[i][j]);
      rowX.push(tx);
      rowY.push(ty);
    }
    outX.push(rowX);
    outY.push(rowY);
  }
  return { warpX: outX, warpY: outY };
}
