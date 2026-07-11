let sortedResult = [];
let tesseractWorker = null;

const imageInput = document.getElementById("imageInput");
const processBtn = document.getElementById("processBtn");
const downloadJsonBtn = document.getElementById("downloadJsonBtn");
const downloadPngBtn = document.getElementById("downloadPngBtn");
const statusEl = document.getElementById("status");
const resultStrip = document.getElementById("resultStrip");

const maxCountInput = document.getElementById("maxCount");
const satThresholdInput = document.getElementById("satThreshold");
const valThresholdInput = document.getElementById("valThreshold");
const minAreaRatioInput = document.getElementById("minAreaRatio");
const labelBandRatioInput = document.getElementById("labelBandRatio");

const workCanvas = document.getElementById("workCanvas");
const previewCanvas = document.getElementById("previewCanvas");

function setStatus(text) {
  statusEl.textContent = text;
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function rgbToLab(rgb) {
  let [r, g, b] = rgb.map((v) => v / 255);

  const gammaExpand = (c) => (c > 0.04045 ? ((c + 0.055) / 1.055) ** 2.4 : c / 12.92);
  r = gammaExpand(r);
  g = gammaExpand(g);
  b = gammaExpand(b);

  let x = r * 0.4124 + g * 0.3576 + b * 0.1805;
  let y = r * 0.2126 + g * 0.7152 + b * 0.0722;
  let z = r * 0.0193 + g * 0.1192 + b * 0.9505;

  x /= 0.95047;
  y /= 1.0;
  z /= 1.08883;

  const f = (t) => (t > 0.008856 ? t ** (1 / 3) : 7.787 * t + 16 / 116);
  const fx = f(x);
  const fy = f(y);
  const fz = f(z);

  return [(116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function dist3(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function nearestNeighborOrder(labs) {
  let start = 0;
  let bestA = -Infinity;
  for (let i = 0; i < labs.length; i += 1) {
    if (labs[i][1] > bestA) {
      bestA = labs[i][1];
      start = i;
    }
  }

  const unvisited = new Set(labs.map((_, i) => i));
  const order = [start];
  unvisited.delete(start);

  while (unvisited.size > 0) {
    const last = order[order.length - 1];
    let nearest = null;
    let nearestDist = Infinity;
    for (const idx of unvisited) {
      const d = dist3(labs[last], labs[idx]);
      if (d < nearestDist) {
        nearestDist = d;
        nearest = idx;
      }
    }
    order.push(nearest);
    unvisited.delete(nearest);
  }

  return order;
}

function orderLength(order, labs) {
  let total = 0;
  for (let i = 1; i < order.length; i += 1) {
    total += dist3(labs[order[i - 1]], labs[order[i]]);
  }
  return total;
}

function twoOpt(order, labs, passes = 4) {
  if (order.length < 4) {
    return order.slice();
  }

  let best = order.slice();
  let bestLen = orderLength(best, labs);

  for (let pass = 0; pass < passes; pass += 1) {
    let improved = false;

    for (let i = 1; i < best.length - 2; i += 1) {
      for (let j = i + 1; j < best.length - 1; j += 1) {
        const candidate = best.slice(0, i).concat(best.slice(i, j + 1).reverse(), best.slice(j + 1));
        const candLen = orderLength(candidate, labs);
        if (candLen + 1e-6 < bestLen) {
          best = candidate;
          bestLen = candLen;
          improved = true;
        }
      }
    }

    if (!improved) {
      break;
    }
  }

  return best;
}

function rgbToHsv01(r, g, b) {
  const rr = r / 255;
  const gg = g / 255;
  const bb = b / 255;
  const max = Math.max(rr, gg, bb);
  const min = Math.min(rr, gg, bb);
  const d = max - min;

  let h = 0;
  if (d > 1e-8) {
    if (max === rr) {
      h = ((gg - bb) / d) % 6;
    } else if (max === gg) {
      h = (bb - rr) / d + 2;
    } else {
      h = (rr - gg) / d + 4;
    }
    h /= 6;
    if (h < 0) {
      h += 1;
    }
  }

  const s = max > 1e-8 ? d / max : 0;
  const v = max;
  return [h, s, v];
}

function connectedComponents(mask, width, height, minPixels) {
  const visited = new Uint8Array(mask.length);
  const boxes = [];
  const queueX = new Int32Array(mask.length);
  const queueY = new Int32Array(mask.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      if (mask[idx] === 0 || visited[idx] === 1) {
        continue;
      }

      let head = 0;
      let tail = 0;
      queueX[tail] = x;
      queueY[tail] = y;
      tail += 1;
      visited[idx] = 1;

      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let count = 0;
      const points = [];

      while (head < tail) {
        const cx = queueX[head];
        const cy = queueY[head];
        head += 1;

        count += 1;
        points.push(cy * width + cx);

        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        const neighbors = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1],
        ];

        for (let i = 0; i < neighbors.length; i += 1) {
          const nx = neighbors[i][0];
          const ny = neighbors[i][1];
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            continue;
          }
          const nIdx = ny * width + nx;
          if (mask[nIdx] === 1 && visited[nIdx] === 0) {
            visited[nIdx] = 1;
            queueX[tail] = nx;
            queueY[tail] = ny;
            tail += 1;
          }
        }
      }

      if (count >= minPixels) {
        boxes.push({ minX, maxX, minY, maxY, count, points });
      }
    }
  }

  return boxes;
}

async function ensureWorker() {
  if (tesseractWorker) {
    return tesseractWorker;
  }

  setStatus("Loading OCR engine...");
  tesseractWorker = await Tesseract.createWorker("eng", 1, {
    workerPath: chrome.runtime.getURL("vendor/worker.min.js"),
    corePath: chrome.runtime.getURL("vendor/tesseract-core.wasm.js"),
    logger: () => {},
  });
  await tesseractWorker.setParameters({
    tessedit_char_whitelist: "0123456789",
  });
  return tesseractWorker;
}

async function readLabelCode(cropCanvas) {
  const worker = await ensureWorker();
  const { data } = await worker.recognize(cropCanvas);
  const text = (data && data.text ? data.text : "").replace(/\s+/g, "");
  const exact = text.match(/\d{3}/);
  if (exact) {
    return exact[0];
  }
  const fallback = text.match(/\d+/);
  if (fallback) {
    return fallback[0].slice(-3).padStart(3, "0");
  }
  return null;
}

function medianFromPoints(imageData, points) {
  const rs = [];
  const gs = [];
  const bs = [];
  const data = imageData.data;
  for (let i = 0; i < points.length; i += 1) {
    const px = points[i] * 4;
    rs.push(data[px]);
    gs.push(data[px + 1]);
    bs.push(data[px + 2]);
  }

  rs.sort((a, b) => a - b);
  gs.sort((a, b) => a - b);
  bs.sort((a, b) => a - b);

  const mid = Math.floor(rs.length / 2);
  return [rs[mid], gs[mid], bs[mid]];
}

function createLabelCrop(canvas, bbox, ratio) {
  const [x, y, w, h] = bbox;
  const bandH = Math.max(8, Math.floor(h * ratio));
  const pad = Math.max(2, Math.floor(w * 0.12));
  const sx = clamp(x - pad, 0, canvas.width - 1);
  const sy = clamp(y + h, 0, canvas.height - 1);
  const sw = clamp(w + pad * 2, 1, canvas.width - sx);
  const sh = clamp(bandH, 1, canvas.height - sy);

  const c = document.createElement("canvas");
  c.width = sw;
  c.height = sh;
  const ctx = c.getContext("2d");
  ctx.drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh);

  const img = ctx.getImageData(0, 0, sw, sh);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
    const bw = gray < 160 ? 0 : 255;
    d[i] = bw;
    d[i + 1] = bw;
    d[i + 2] = bw;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

async function processImage(file) {
  const satThreshold = clamp(parseFloat(satThresholdInput.value) || 0.18, 0, 1);
  const valThreshold = clamp(parseFloat(valThresholdInput.value) || 0.15, 0, 1);
  const minAreaRatio = clamp(parseFloat(minAreaRatioInput.value) || 0.0006, 0.00001, 0.2);
  const labelBandRatio = clamp(parseFloat(labelBandRatioInput.value) || 0.75, 0.1, 2);
  const maxCount = Math.max(0, parseInt(maxCountInput.value, 10) || 0);

  const bitmap = await createImageBitmap(file);
  workCanvas.width = bitmap.width;
  workCanvas.height = bitmap.height;
  const ctx = workCanvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);

  const imageData = ctx.getImageData(0, 0, workCanvas.width, workCanvas.height);
  const { data } = imageData;
  const mask = new Uint8Array(workCanvas.width * workCanvas.height);

  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    const [h, s, v] = rgbToHsv01(data[i], data[i + 1], data[i + 2]);
    mask[p] = s >= satThreshold && v >= valThreshold ? 1 : 0;
  }

  const minPixels = Math.max(12, Math.floor(minAreaRatio * workCanvas.width * workCanvas.height));
  const components = connectedComponents(mask, workCanvas.width, workCanvas.height, minPixels)
    .sort((a, b) => b.count - a.count);

  if (components.length < 2) {
    throw new Error("Could not detect enough swatches. Try lowering thresholds.");
  }

  const limited = maxCount > 0 ? components.slice(0, maxCount) : components;
  const swatches = [];

  for (let i = 0; i < limited.length; i += 1) {
    const comp = limited[i];
    const w = comp.maxX - comp.minX + 1;
    const h = comp.maxY - comp.minY + 1;
    const rgb = medianFromPoints(imageData, comp.points);
    const bbox = [comp.minX, comp.minY, w, h];

    setStatus(`Reading labels ${i + 1}/${limited.length}...`);
    const cropCanvas = createLabelCrop(workCanvas, bbox, labelBandRatio);
    let code = null;
    try {
      code = await readLabelCode(cropCanvas);
    } catch (err) {
      code = null;
    }
    if (!code) {
      code = `UNK${String(i + 1).padStart(3, "0")}`;
    }

    swatches.push({
      code,
      rgb,
      pixelCount: comp.count,
      bbox,
    });
  }

  const labs = swatches.map((s) => rgbToLab(s.rgb));
  const initialOrder = nearestNeighborOrder(labs);
  const bestOrder = twoOpt(initialOrder, labs);

  const sorted = bestOrder.map((idx, i) => {
    const s = swatches[idx];
    return {
      order: i + 1,
      code: s.code,
      rgb: s.rgb,
      hex: `#${s.rgb.map((v) => v.toString(16).padStart(2, "0")).join("").toUpperCase()}`,
      pixels: s.pixelCount,
      bbox: s.bbox,
    };
  });

  const used = new Set();
  for (let i = 0; i < sorted.length; i += 1) {
    let code = sorted[i].code;
    if (!used.has(code)) {
      used.add(code);
      continue;
    }
    let n = 2;
    while (used.has(`${code}_${n}`)) {
      n += 1;
    }
    sorted[i].code = `${code}_${n}`;
    used.add(sorted[i].code);
  }

  return sorted;
}

function renderSwatches(items) {
  resultStrip.innerHTML = "";
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    const card = document.createElement("div");
    card.className = "swatch";

    const colorBlock = document.createElement("div");
    colorBlock.className = "swatch-color";
    colorBlock.style.background = item.hex;

    const codeEl = document.createElement("div");
    codeEl.className = "swatch-code";
    codeEl.textContent = item.code;

    const hexEl = document.createElement("div");
    hexEl.className = "swatch-hex";
    hexEl.textContent = item.hex;

    card.appendChild(colorBlock);
    card.appendChild(codeEl);
    card.appendChild(hexEl);
    resultStrip.appendChild(card);
  }
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1200);
}

function createPreviewPng(items) {
  const blockW = 98;
  const blockH = 118;
  previewCanvas.width = Math.max(1, items.length * blockW);
  previewCanvas.height = blockH;
  const ctx = previewCanvas.getContext("2d");

  for (let i = 0; i < items.length; i += 1) {
    ctx.fillStyle = items[i].hex;
    ctx.fillRect(i * blockW, 0, blockW, blockH - 24);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(i * blockW, blockH - 24, blockW, 24);
    ctx.fillStyle = "#1c1c1c";
    ctx.font = "bold 12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(items[i].code, i * blockW + blockW / 2, blockH - 8);
  }

  return new Promise((resolve) => {
    previewCanvas.toBlob((blob) => resolve(blob), "image/png");
  });
}

processBtn.addEventListener("click", async () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) {
    setStatus("Please choose an image first.");
    return;
  }

  processBtn.disabled = true;
  downloadJsonBtn.disabled = true;
  downloadPngBtn.disabled = true;

  try {
    setStatus("Detecting swatches...");
    sortedResult = await processImage(file);
    renderSwatches(sortedResult);

    setStatus(`Done. Found ${sortedResult.length} swatches.`);
    downloadJsonBtn.disabled = false;
    downloadPngBtn.disabled = false;
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    sortedResult = [];
    resultStrip.innerHTML = "";
  } finally {
    processBtn.disabled = false;
  }
});

downloadJsonBtn.addEventListener("click", () => {
  if (sortedResult.length === 0) {
    return;
  }
  const blob = new Blob([JSON.stringify(sortedResult, null, 2)], { type: "application/json" });
  downloadBlob("sorted_markers.json", blob);
});

downloadPngBtn.addEventListener("click", async () => {
  if (sortedResult.length === 0) {
    return;
  }
  const blob = await createPreviewPng(sortedResult);
  if (blob) {
    downloadBlob("sorted_markers_preview.png", blob);
  }
});

window.addEventListener("unload", async () => {
  if (tesseractWorker) {
    try {
      await tesseractWorker.terminate();
    } catch (err) {
      // ignore
    }
    tesseractWorker = null;
  }
});
