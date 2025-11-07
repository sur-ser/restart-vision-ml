

# restart-vision-ml

Document (Receipt / Document / Screenshot) detection using **YOLOv8 (ONNXRuntime)** with a **deterministic Sharp pre-processing pipeline** and a **post‑refine** stage to pick the main class, validate status/navigation bars (without mutating their boxes), and synthesize a consistent `primaryBox`.

> Node.js ≥ **20.10**, ESM-only package (`"type": "module"`). CPU inference by default.

---

## Features

- **ONNXRuntime (CPU) inference** for an exported YOLOv8 model.
- **Deterministic letterbox pre-processing** (Sharp): resize → letterbox (value 114) → CHW float32 [0..1].
- **Per-class NMS** with IoU threshold (configurable).
- **Post-refine**:
  - main class selection among `Document`, `Receipt`, `Screenshot` (+ tie-breakers by aspect ratio or bars);
  - non-mutating **Top status bar** / **Bottom nav bar** validation inside the screenshot container;
  - `primaryBox` synthesis and replacement if IoU is too low;
  - **Unknown** fallback (synthetic full-frame detection);
  - optional diagnostics collection.
- **Image utilities**: metadata (sRGB), color quantization (top-5 buckets), edge intensity, skew estimation, deskew.
- **Drawing** utilities: render detection boxes and labels onto the image.
- **TypeScript-first** codebase with strict typing.

---

## Requirements

- Node.js ≥ **20.10**
- OS packages: none special for most platforms (Sharp bundles libvips).
- A YOLOv8 **ONNX** model and a matching `classes.txt` (one class name per line).

> Large models should not be committed to Git — use Git LFS for `*.onnx`, `*.pt`, `*.tflite`.

---

## Installation

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/restart-vision-ml.git
cd restart-vision-ml
npm i
```

---

## Model placement

By default the detector looks for:
```
./models/yolo/800-50/best.onnx
./models/yolo/800-50/classes.txt
```

You can override this via the detector constructor:

```ts
import { YoloV8OnnxDetector } from "./src/detectors/yolo-v8-onnx.detector.js";

const detector = new YoloV8OnnxDetector({
  modelPath: "./models/my-yolo/best.onnx",
  inputSize: 800,
  confidenceThreshold: 0.10,
  iouThreshold: 0.45,
  classNames: [ "Receipt", "Document", "Screenshot", "Top status bar", "Bottom nav bar" ],
});
```

---

## Quick start

```ts
// demo.ts
import { DocumentAnalyzer } from "./src/analyzer/document-analyzer.js";
import { drawDetections } from "./src/utils/draw-detections.js";

const imgPath = process.argv[2];
if (!imgPath) {
  console.error("Usage: npm run dev -- path/to/image.jpg");
  process.exit(1);
}

const analyzer = new DocumentAnalyzer();
const result = await analyzer.analyze(imgPath, {
  refine: true, // run post-refine
});

console.log(JSON.stringify(result, null, 2));

// Optional: visualize detections
const { outPath } = await drawDetections(imgPath, result.detections, "./output");
console.log("Saved overlay:", outPath);
```

Run:

```bash
npm run dev -- path/to/image.jpg
```

---

## Public API (high level)

### `DocumentAnalyzer`

```ts
import { DocumentAnalyzer } from "./src/analyzer/document-analyzer.js";

const analyzer = new DocumentAnalyzer(); // accepts optional injected ImageProcessor / Detector
const result = await analyzer.analyze(src, {
  refine: true,
  refineOptions: {
    tol: {
      minBarWidthFrac: 0.8,
      topBottomBandFrac: 0.12,
      maxBarHeightFrac: 0.15,
      insideSlackPx: 8,
      classMargin: 0.07,
      minPrimaryAreaFrac: 0.20,
      screenshotMinCoverFrac: 0.80,
      screenshotBarBonus: 0.05,
      screenshotBothBarsBonus: 0.03,
      screenshotCoverBonus: 0.05,
      altCandidateMaxDelta: 0.06
    },
    allowRetype: true,
    keepDiagnostics: true
  }
});
```

### `refineDocumentAnalysis(result, options?)`

Post-processes a raw `DocumentAnalysisResult` produced by `DocumentAnalyzer`.  
Does **not** mutate incoming bar boxes; uses container intersection for metrics only.  
May retype main class, update `primaryBox`, add synthetic main-class detection, and collect diagnostics.

---

## Public API (building blocks)

### `YoloV8OnnxDetector`

- **Constructor options**
  - `modelPath?: string` — path to ONNX file (default `./models/yolo/800-50/best.onnx`)
  - `inputSize?: number` — square side, default `800`
  - `confidenceThreshold?: number` — default `0.10`
  - `iouThreshold?: number` — default `0.45`
  - `classNames?: string[]` — if omitted, detector will attempt to read `classes.txt` next to the model
- **Methods**
  - `isReady(): boolean`
  - `initialize(): Promise<void>`
  - `detectObjects(src): Promise<{ detections, originalBuffer, width, height }>`
  - `classNames: string[]` (getter)

### `ImageProcessor`

- `getImageMetadata(src)`
- `getImageBuffer(src)`
- `getImageAsBase64(src)`
- `analyzeImageColors(src)` → top-5 `rgb(r,g,b)` buckets
- `detectEdges(src)` → mean gradient based score `[0..1]`
- `estimateSkewAngleSharp(src, opts?)`
- `deskewWithAngle(src, angleDeg)`

### Utils

- `drawDetections(imagePath, detections, outDir, palette?)`
- `drawDetectionsOnBuffer(imageBuffer, detections, palette?)`
- `cropByDetection(imageBuffer, det)`

---

## Data types

The primary result type:

```ts
import type { DocumentAnalysisResult } from "./src/interfaces/document-analysis-result.js";

/**
 * {
 *   source, timestamp, meta,
 *   summary: {
 *     documentType, confidence, background, primaryBox, quality { ... }
 *   },
 *   detections: Array<{ label, score, box, meta? }>,
 *   signals?: { colors?: string[], diagnostics?: string[], classificationTopK?: ... },
 *   model?: { detector?: string; classes?: string[]; inputSize?: number; thresholds?: { conf?: number; iou?: number } }
 * }
 */
```

Main enum for the summary:
```ts
export enum DocumentType {
  RECEIPT = "Receipt",
  DOCUMENT = "Document",
  SCREENSHOT = "Screenshot",
  UNKNOWN = "Unknown"
}
```

---

## Project structure

```
src/
  analyzer/
    document-analyzer.ts
  analysis/
    analysis-refine.ts
  detectors/
    yolo-v8-onnx.detector.ts
  processors/
    image-processor.ts
  interfaces/
    document-analysis-result.ts
  types/
    detection.types.ts
    background-type.enum.ts
    document-type.enum.ts
    image-source.ts
  utils/
    open-sharp.ts
    draw-detections.ts
    crop-by-detection.ts
    deskew.ts
models/
  yolo/...
output/
```

---

## Scripts

- `npm run dev` — run demo script (`src/demo.ts`).
- `npm run build` — compile TypeScript to `dist/` and emit types.
- `npm start` — run built demo (`dist/demo.js`).
- `npm run lint` — ESLint.
- `npm run format` — Prettier.
- `npm run clean` — remove `dist/`.

---

## Troubleshooting

- **Node version**: ensure `node -v` ≥ `20.10`.
- **ONNX model path**: if you see *"YOLO model not found"* — check `modelPath` and working directory.
- **Missing classes**: if `classes.txt` is absent and `classNames` not provided — detector will throw.
- **Sharp install**: if prebuilt binaries are missing, reinstall with `npm rebuild sharp`.
- **Scores look ~0.5 after sigmoid**: **do not** apply sigmoid — the exported model already emits normalized scores for classes (the post-process expects raw values).
- **Do not install core Node modules**: never `npm i node:path fs url`. They come from Node.

---

## License

ISC © YOUR_NAME