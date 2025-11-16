import path from "node:path";
import { promises as fs } from "node:fs";
import sharp from "sharp";
import {drawDetectionsOnBuffer} from "../src/utils/draw-detections";
import {YoloV8OnnxDetector} from "../src/detectors/yolo-v8-onnx.detector";
import {cropByDetection} from "../src/utils/crop-by-detection";

async function run() {
    const [, , imagePathArg] = process.argv;
    if (!imagePathArg) {
        console.error("Usage: tsx examples/raw-detect.ts path/to/image.jpg");
        process.exit(1);
    }
    const imagePath = path.resolve(process.cwd(), imagePathArg);
    const outDir = path.resolve(process.cwd(), "images/processed");
    await fs.mkdir(outDir, { recursive: true });

    const detector = new YoloV8OnnxDetector({
        modelPath: path.resolve(process.cwd(), "models/yolo/prod_yolov8s_2025-11-10/last.onnx"),
    });
    if (!detector.isReady()) {
        await detector.initialize();
    }

    const { detections, originalBuffer } = await detector.detectObjects(imagePath);

// оставим по одному боксу на класс (максимальный score)
    const bestPerClass = Array.from(
        detections.reduce((m, d) => (!m.has(d.label) || (d.score ?? 0) > (m.get(d.label)!.score ?? 0) ? m.set(d.label, d) : m), new Map<string, typeof detections[0]>()).values()
    );

    console.log("Detections:", bestPerClass.map(d => ({label: d.label, score: d.score})));

    const overlay = await drawDetectionsOnBuffer(originalBuffer, bestPerClass, {
        Receipt: "#22c55e",
        Screenshot: "#3b82f6",
        Document: "#f97316",
        "Bottom nav bar": "#a855f7",
        "Top status bar": "#10b981",
    });
    const overlayPath = path.join(outDir, path.basename(imagePath));
    await sharp(overlay).toFile(overlayPath);
    console.log("Saved overlay:", overlayPath);

// crop первого бокса (если есть)
    if (bestPerClass[0]) {
        const crop = await cropByDetection(originalBuffer, bestPerClass[0]);
        const cropPath = path.join(outDir, `crop-${path.basename(imagePath)}`);
        await sharp(crop).toFile(cropPath);
        console.log("Saved crop:", cropPath);
    }
}

run().catch(console.error);