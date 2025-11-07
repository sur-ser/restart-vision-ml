import { YoloV8OnnxDetector } from "../detectors/yolo-v8-onnx.detector";
import { ImageProcessor } from "../processors/image-processor";
import { DocumentType } from "../types/document-type.enum";
import { BackgroundType } from "../types/background-type.enum";
import path from "path";
import {DocumentAnalysisResult} from "../interfaces/document-analysis-result";
import {refineDocumentAnalysis} from "./analysis-refine";
type AnalyzeOptions = {
    refine?: boolean;      // If true — run the post-processor (refine step)
    refineOptions?: Parameters<typeof refineDocumentAnalysis>[1];
};
export class DocumentAnalyzer {
    constructor(
        private readonly imageProcessor = new ImageProcessor(),
        private readonly detector = new YoloV8OnnxDetector()
    ) {}

    async analyze(src: string | Buffer, opts?: AnalyzeOptions): Promise<DocumentAnalysisResult> {

        if (!this.detector.isReady?.() && (this.detector as any).initialize) {
            await (this.detector as any).initialize();
        }

        const isBuffer = Buffer.isBuffer(src);
        const sourcePath = !isBuffer ? src as string : undefined;
        const filename = sourcePath ? path.basename(sourcePath) : undefined;

        // Extract base image metadata and signals
        const meta = await this.imageProcessor.getImageMetadata(src);
        const colors = await this.imageProcessor.analyzeImageColors(src);
        const edgeIntensity = await this.imageProcessor.detectEdges(src);

        // Run YOLO detection
        const { detections } = await this.detector.detectObjects(src);

        // Pick the top detection and derive the document type
        const top = detections.length
            ? detections.reduce((best, cur) => (cur.score > best.score ? cur : best))
            : null;

        const documentType = this.mapLabelToType(top?.label);
        const confidence = top?.score ?? 0;
        const primaryBox = top?.box ?? undefined;

        const aspectRatio = meta.width && meta.height ? meta.width / meta.height : 1;
        const clarity: 'low' | 'medium' | 'high' =
            edgeIntensity < 0.01 ? "low" :
                edgeIntensity > 0.05 ? "high" : "medium";

        // Build the final result object
        const result: DocumentAnalysisResult = {
            source: {
                filename,
                path: sourcePath,
            },
            timestamp: new Date().toISOString(),
            meta: {
                width: meta.width,
                height: meta.height,
                aspectRatio,
                format: meta.format,
                colorspace: meta.colorspace,
                density: meta.density,
            },
            summary: {
                documentType,
                confidence,
                background: BackgroundType.UNKNOWN,
                primaryBox,
                quality: {
                    edgeIntensity,
                    clarity,
                    readabilityScore: Math.min(edgeIntensity * 10, 1),
                    isPartial: false,
                },
            },
            detections,
            signals: {
                colors,
            },
            model: {
                detector: "yolov8-onnx",
                classes: this.detector.classNames ?? [],
                inputSize: 800,
                thresholds: {
                    conf: 0.1,
                    iou: 0.45,
                },
            },
        };
        if (opts?.refine) {
            return refineDocumentAnalysis(result, opts.refineOptions);
        }
        return result;
    }

    /** Maps a detection label to a DocumentType. */
    private mapLabelToType(label?: string): DocumentType {
        if (!label) return DocumentType.UNKNOWN;
        const l = label.toLowerCase();
        if (l.includes("receipt")) return DocumentType.RECEIPT;
        if (l.includes("document")) return DocumentType.DOCUMENT;
        if (l.includes("screenshot") || l.includes("bar"))
            return DocumentType.SCREENSHOT;
        return DocumentType.UNKNOWN;
    }
}