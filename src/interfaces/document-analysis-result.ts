import type { DetectionAxisAligned } from '../types/detection.types';
import {DocumentType} from "../types/document-type.enum";
import {BackgroundType} from "../types/background-type.enum";

export interface AnalysisSignals {
    colors?: string[]; // e.g., ['rgb(248,248,248)']
    classificationTopK?: { label: string; score: number }[]; // Top‑K results from zero‑shot classification (if available)
    diagnostics?: string[];
}
export interface DocumentAnalysisSummary {
    documentType: DocumentType;  // Receipt | Document | Screenshot | ...
    confidence?: number; // Model confidence for documentType
    background: BackgroundType;
    primaryBox?: { x1:number; y1:number; x2:number; y2:number }; // Primary document bounding box (if available)
    primarySynthetic?: boolean;         // True if the primary box was synthesized by logic (not predicted by the model)
    primaryBoxSource?: 'detector'|'refine_best'|'refine_fallback'; // Origin of the primary box: produced by detector, overridden by refine, or fallback
    quality: {
        edgeIntensity?: number;       // Mean edge magnitude computed by ImageProcessor
        readabilityScore?: number;    // Heuristic readability score in [0..1]
        clarity?: 'low' | 'medium' | 'high';
        skewAngleDeg?: number;        // Estimated skew angle in degrees
        isPartial?: boolean;          // True if the document is not fully in frame
    };
}
export interface DocumentAnalysisResult {
    // 1) Откуда картинка
    source: {
        filename?: string;          // Original filename, if known
        path?: string;              // File path, if known
    };

    timestamp: string;            // ISO 8601 timestamp

    // 2) Метаданные изображения
    meta: {
        width: number;
        height: number;
        aspectRatio: number;        // width / height
        format: string;             // 'jpeg' | 'png' | ...
        colorspace: string;         // 'srgb' | ...
        density?: number;           // Dots per inch (DPI), if available
    };

    // 3) Сводка по документу (то, что обычно важно потребителю)
    summary: DocumentAnalysisSummary;

    // 4) Детекции от YOLO (все боксы)
    detections: DetectionAxisAligned[]; // Raw detector boxes: { label, score, box }

    // 5) “Сырые” сигналы анализа (для дебага/обогащения UI)
    signals?: AnalysisSignals; // Raw analysis signals for debugging/visualization

    // 6) Model/technical info for transparency and reproducibility
    model?: {
        detector?: 'yolov8-onnx' | string;
        classes?: string[];             // Class order used by the detector
        inputSize?: number;             // e.g., 800
        thresholds?: { conf?: number; iou?: number }; // Detector thresholds used at inference
    };
}