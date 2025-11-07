import path from "node:path";
import { promises as fs } from "node:fs";
import sharp from "sharp";
import {DocumentAnalyzer} from "../src/analysis/document-analyzer";
import {ImageProcessor} from "../src/processors/image-processor";
import {YoloV8OnnxDetector} from "../src/detectors/yolo-v8-onnx.detector";
import {openSharp} from "../src/utils/open-sharp";
import {drawDetectionsOnBuffer} from "../src/utils/draw-detections";



const [, , imagePathArg] = process.argv;
if (!imagePathArg) {
    console.error("Usage: tsx examples/01-analyze-one.ts path/to/image.jpg");
    process.exit(1);
}
const imagePath = path.resolve(process.cwd(), imagePathArg);
const outDir = path.resolve(process.cwd(), "images/processed");
await fs.mkdir(outDir, { recursive: true });

const analyzer = new DocumentAnalyzer(
    new ImageProcessor(),
    new YoloV8OnnxDetector({
        modelPath: path.resolve(process.cwd(), "models/yolo/800-50/best.onnx"),
    })
);

const result = await analyzer.analyze(imagePath, { refine: true });
console.log("Summary:", JSON.stringify(result.summary, null, 2));

const { sh } = openSharp(imagePath);
const buffer = await sh.toBuffer();
const palette: Record<string, string> = {
    Receipt: "#22c55e",
    Screenshot: "#3b82f6",
    Document: "#f97316",
    "Bottom nav bar": "#a855f7",
    "Top status bar": "#10b981",
};
const vis = await drawDetectionsOnBuffer(buffer, result.detections ?? [], palette);
const outPath = path.join(outDir, path.basename(imagePath));
await sharp(vis).toFile(outPath);
console.log("Saved overlay:", outPath);