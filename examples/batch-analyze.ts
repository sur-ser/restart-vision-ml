import path from "node:path";
import { promises as fs } from "node:fs";
import sharp from "sharp";
import {DocumentAnalyzer} from "../src/analysis/document-analyzer";
import {ImageProcessor} from "../src/processors/image-processor";
import {YoloV8OnnxDetector} from "../src/detectors/yolo-v8-onnx.detector";
import {drawDetectionsOnBuffer} from "../src/utils/draw-detections";
import {openSharp} from "../src/utils/open-sharp";
import {DocumentType} from "../src/types/document-type.enum";


const [, , inputDirArg] = process.argv;
if (!inputDirArg) {
    console.error("Usage: tsx examples/02-batch-analyze.ts path/to/images_dir");
    process.exit(1);
}
const inputDir = path.resolve(process.cwd(), inputDirArg);
const outDir = path.resolve(process.cwd(), "images/processed");
await fs.mkdir(outDir, { recursive: true });

const analyzer = new DocumentAnalyzer(
    new ImageProcessor(),
    new YoloV8OnnxDetector({
        modelPath: path.resolve(process.cwd(), "models/yolo/800-50/best.onnx"),
    })
);

const files = await fs.readdir(inputDir);
const imageFiles = files.filter((f) => /\.(jpe?g|png|webp|tiff)$/i.test(f));
console.log(`🖼 Found ${imageFiles.length} images`);

const resume: Record<DocumentType, number> = {
    [DocumentType.RECEIPT]: 0,
    [DocumentType.DOCUMENT]: 0,
    [DocumentType.SCREENSHOT]: 0,
    [DocumentType.UNKNOWN]: 0,
};

const allResults: Array<{
    filename: string;
    inputPath: string;
    outputPath?: string;
    summary: any;
    detections: any[];
    signals?: any;
    error?: string;
}> = [];

for (const filename of imageFiles) {
    const inputPath = path.join(inputDir, filename);
    const outputPath = path.join(outDir, filename);

    try {
        const result = await analyzer.analyze(inputPath, { refine: true });
        const type = result.summary.documentType;
        resume[type] = (resume[type] ?? 0) + 1;

        let wrotePreview = false;
        if (result.detections?.length) {
            const { sh } = openSharp(inputPath);
            const buffer = await sh.toBuffer();
            const vis = await drawDetectionsOnBuffer(buffer, result.detections);
            await sharp(vis).toFile(outputPath);
            wrotePreview = true;
        }

        allResults.push({
            filename,
            inputPath,
            outputPath: wrotePreview ? outputPath : undefined,
            summary: result.summary,
            detections: result.detections ?? [],
            signals: result.signals,
        });

        const conf = ((result.summary.confidence ?? 0) * 100).toFixed(1);
        console.log(`✔ ${filename}: ${type} (${conf}%)`);
    } catch (e) {
        const message = (e as Error)?.message || String(e);
        console.error(`❌ ${filename}: ${message}`);
        allResults.push({
            filename,
            inputPath,
            summary: { documentType: DocumentType.UNKNOWN, confidence: 0 },
            detections: [],
            error: message,
        });
    }
}

const resultJsonPath = path.join(outDir, "result.json");
await fs.writeFile(
    resultJsonPath,
    JSON.stringify(
        {
            timestamp: new Date().toISOString(),
            inputDir: inputDir,
            total: imageFiles.length,
            resume,
            results: allResults,
        },
        null,
        2
    ),
    "utf8"
);
console.log(`\n💾 Report saved: ${resultJsonPath}`);