import sharp from "sharp";
import { promises as fs } from "fs";
import * as syncFs from "fs";
import path from "path";
import ort from 'onnxruntime-node';
import {ImageProcessor} from "../processors/image-processor";
import {DetectionAxisAligned} from "../types/detection.types";
import {ImageSource} from "../types/image-source";
import {IDetector} from "../interfaces/detector.interface";

/**
 * YOLOv8 ONNX Detector
 *
 * IMPORTANT: This model outputs raw class logits without applying sigmoid.
 * The scores are already in the 0–1 range after export, but they are raw scores,
 * not probabilities. Do NOT apply sigmoid to these normalized scores.
 */
export class YoloV8OnnxDetector implements IDetector {
    private modelPath: string = './models/yolo/800-50/best.onnx';
    private imageProcessor: ImageProcessor;
    private yoloSession: any = null;
    private INPUT_SIZE: number = 800;
    private CONFIDENCE_THRESHOLD: number = 0.1;
    private IOU_THRESHOLD: number = 0.45;
    private readonly debug = true;
    private CLASS_NAMES: string[] = [];

    constructor(options?: {
        modelPath?: string;
        inputSize?: number;
        confidenceThreshold?: number;
        iouThreshold?: number;
        classNames?: string[];
    }) {
        this.modelPath = options?.modelPath || this.modelPath;
        this.INPUT_SIZE = options?.inputSize || this.INPUT_SIZE;
        this.CONFIDENCE_THRESHOLD = options?.confidenceThreshold || this.CONFIDENCE_THRESHOLD;
        this.IOU_THRESHOLD = options?.iouThreshold || this.IOU_THRESHOLD;
        this.CLASS_NAMES = options?.classNames || this.CLASS_NAMES;

        if(this.CLASS_NAMES.length === 0){
            this.detectAndReadClassesTxt();
            if(this.CLASS_NAMES.length === 0){
                throw Error('Список классов пуст. Положи classes.txt рядом с моделью или передай classNames в конструктор.');
            }
        }

        this.imageProcessor = new ImageProcessor();
    }

    isReady(): boolean { return !!this.yoloSession; }
    public get classNames() { return this.CLASS_NAMES; }

    detectAndReadClassesTxt() {
        const modelDir = this.modelPath.split('/').slice(0, -1).join('/');
        const classesPath = path.join(modelDir, 'classes.txt');
        if (!syncFs.existsSync(classesPath)) {
            this.log(`classes.txt не найден по пути: ${classesPath}`);
            return;
        }
        try {
            const lines = syncFs.readFileSync(classesPath, 'utf8').split(/\r?\n/).map(s => s.trim()).filter(Boolean);
            this.CLASS_NAMES = lines;
            this.log(`Классы загружены: ${this.CLASS_NAMES.length} шт`);
        } catch (err) {
            console.error('Ошибка при чтении classes.txt:', err);
        }
    }

    log(text: string) {
        if (this.debug) {
            console.log(text);
        }
    }

    async initialize(): Promise<void> {
        this.log("Инициализация YOLO классификатора...");
        this.log(`📄 Загрузка модели: ${this.modelPath}`);

        const fileExists = await fs.access(this.modelPath).then(() => true, () => false);
        if (!fileExists) {
            throw new Error(`YOLO модель не найдена: ${this.modelPath}`);
        }

        try {
            this.yoloSession = await (ort as any).InferenceSession.create(this.modelPath);
            this.log('YOLO модель успешно загружена');
            this.log(`   Входные тензоры: ${JSON.stringify(this.yoloSession.inputNames)}`);
            this.log(`   Выходные тензоры: ${JSON.stringify(this.yoloSession.outputNames)}`);
        } catch (e) {
            console.error("Ошибка при загрузке YOLO модели:", e);
            throw e;
        }
    }

    async detectObjects(src: ImageSource) {
        if(!this.yoloSession){
            throw new Error("YOLO модель не инициализирована");
        }

        const { tensor, originalWidth, originalHeight, scale, padX, padY, originalBuffer } =
            await this.preprocessImage(src);

        const feeds: Record<string, any> = {};
        feeds[this.yoloSession.inputNames[0]] = tensor;

        const results = await this.yoloSession.run(feeds);

        if (this.debug) {
            for (const key in results) {
                const output = results[key];
                this.log(`Результат ${key}: shape = ${output.dims.join('x')}`);

                // Анализ диапазона значений
                const data = output.data as Float32Array;
                const min = Math.min(...Array.from(data.slice(0, 1000))); // первые 1000 для скорости
                const max = Math.max(...Array.from(data.slice(0, 1000)));
                this.log(`   Диапазон значений (sample): [${min.toFixed(2)}, ${max.toFixed(2)}]`);
            }
        }

        const output = this.pickDetOutput(results, this.CLASS_NAMES.length);

        const detections = this.postprocessOutput(
            output,
            originalWidth,
            originalHeight,
            scale,
            padX,
            padY
        );

        return { detections, originalBuffer, width: originalWidth, height: originalHeight };
    }

    private async preprocessImage(src: ImageSource): Promise<{
        tensor: any;
        originalWidth: number;
        originalHeight: number;
        scale: number;
        padX: number;
        padY: number;
        originalBuffer: Buffer;
    }> {
        // 1) Read the original source without any hidden transforms
        let originalBuffer: Buffer;
        let originalWidth = 0, originalHeight = 0;

        if (Buffer.isBuffer(src)) {
            originalBuffer = src;
            const meta = await sharp(src).metadata();
            originalWidth = meta.width || 0;
            originalHeight = meta.height || 0;
        } else {
            // src is a filesystem path
            originalBuffer = await fs.readFile(src as string);
            const meta = await sharp(originalBuffer).metadata();
            originalWidth = meta.width || 0;
            originalHeight = meta.height || 0;
        }

        const T = this.INPUT_SIZE; // 800
        const scale = Math.min(T / originalWidth, T / originalHeight);
        const newW = Math.floor(originalWidth * scale);
        const newH = Math.floor(originalHeight * scale);
        const padX = Math.floor((T - newW) / 2);
        const padY = Math.floor((T - newH) / 2);

        // 2) Resize, convert to sRGB, drop alpha, and get raw RGB (HWC)
        const resized = await sharp(originalBuffer)
            .resize(newW, newH, { fit: 'fill', kernel: 'lanczos3' })
            .toColorspace('srgb')      // ВАЖНО: как в твоём тесте JS
            .removeAlpha()
            .raw()
            .toBuffer();

        // 3) Manual letterbox padding with value 114
        const canvas = Buffer.alloc(T * T * 3, 114);
        for (let y = 0; y < newH; y++) {
            const srcStart = y * newW * 3;
            const dstStart = ((padY + y) * T + padX) * 3;
            resized.copy(canvas, dstStart, srcStart, srcStart + newW * 3);
        }

        // 4) HWC → CHW, scale [0..255] to [0..1]
        const area = T * T;
        const float32Data = new Float32Array(3 * area);
        for (let i = 0; i < area; i++) {
            float32Data[i]           = canvas[i * 3] / 255;       // R
            float32Data[area + i]    = canvas[i * 3 + 1] / 255;   // G
            float32Data[2 * area + i]= canvas[i * 3 + 2] / 255;   // B
        }

        const tensor = new ort.Tensor('float32', float32Data, [1, 3, T, T]);

        // Debug: input checksum (should match the reference test)
        if (this.debug) {
            const sum = float32Data.reduce((a, b) => a + b, 0);
            this.log(`Input checksum: ${sum.toFixed(6)}`);
        }

        return { tensor, originalWidth, originalHeight, scale, padX, padY, originalBuffer };
    }

    private postprocessOutput(
        output: any,
        originalWidth: number,
        originalHeight: number,
        scale: number,
        padX: number,
        padY: number
    ): DetectionAxisAligned[] {
        const parsed = this.parseYoloOnnxOutputCorrect(output, this.CLASS_NAMES.length);

        if (this.debug) {
            this.log(`Детекций до фильтрации: ${parsed.length}`);

            const aboveThreshold = parsed.filter(p => p.confidence > this.CONFIDENCE_THRESHOLD);
            this.log(`Детекций после порога confidence (${this.CONFIDENCE_THRESHOLD}): ${aboveThreshold.length}`);

            // Show top‑5 detections
            const top5 = parsed.slice(0, 5);
            top5.forEach((det, i) => {
                this.log(`  Детекция ${i}: класс=${this.CLASS_NAMES[det.classId]}, confidence=${det.confidence.toFixed(3)}, box=[${det.x.toFixed(1)},${det.y.toFixed(1)},${det.w.toFixed(1)},${det.h.toFixed(1)}]`);
            });
        }

        const filtered = parsed.filter(p => p.confidence > this.CONFIDENCE_THRESHOLD);
        const nmsBoxes = this.applyNMS(filtered);

        if (this.debug) {
            this.log(`Детекций после NMS: ${nmsBoxes.length}`);
        }

        const detections: DetectionAxisAligned[] = nmsBoxes.map((box) => {
            // Map from letterbox coordinates back to original image coordinates
            const x1 = ((box.x - box.w / 2) - padX) / scale;
            const y1 = ((box.y - box.h / 2) - padY) / scale;
            const x2 = ((box.x + box.w / 2) - padX) / scale;
            const y2 = ((box.y + box.h / 2) - padY) / scale;

            return {
                label: this.CLASS_NAMES[box.classId],
                score: box.confidence,
                box: {
                    x1: Math.max(0, Math.round(x1)),
                    y1: Math.max(0, Math.round(y1)),
                    x2: Math.min(originalWidth, Math.round(x2)),
                    y2: Math.min(originalHeight, Math.round(y2)),
                },
            };
        });

        return detections;
    }

    /**
     * Correct parser for this YOLOv8 ONNX output.
     *
     * IMPORTANT: The model emits raw scores. We do NOT apply sigmoid, because that would
     * push scores toward ~0.5. We keep raw scores as‑is and compare them directly,
     * then threshold with `CONFIDENCE_THRESHOLD` and run class‑wise NMS.
     */
    private parseYoloOnnxOutputCorrect(
        output: { data: Float32Array; dims: number[] },
        numClasses: number
    ) {
        const data = output.data as Float32Array;
        const dims = output.dims; // [1, C, N] или [1, N, C]
        const expectedChannels = 4 + numClasses;

        const results: Array<{ x:number; y:number; w:number; h:number; classId:number; confidence:number }> = [];

        let channelFirst = false, N = 0;
        if (dims.length === 3 && dims[0] === 1 && dims[1] === expectedChannels) {
            channelFirst = true; N = dims[2];
        } else if (dims.length === 3 && dims[0] === 1 && dims[2] === expectedChannels) {
            channelFirst = false; N = dims[1];
        } else {
            throw new Error(`Unexpected output shape: ${dims.join('x')}`);
        }

        for (let i = 0; i < N; i++) {
            let x:number, y:number, w:number, h:number;

            if (channelFirst) {
                x = data[0 * N + i];
                y = data[1 * N + i];
                w = data[2 * N + i];
                h = data[3 * N + i];
            } else {
                const base = i * expectedChannels;
                x = data[base + 0];
                y = data[base + 1];
                w = data[base + 2];
                h = data[base + 3];
            }

            if (w <= 0 || h <= 0) continue;

            // берем raw scores БЕЗ sigmoid/нормализаций
            let bestScore = -Infinity, bestClass = -1;
            if (channelFirst) {
                for (let c = 0; c < numClasses; c++) {
                    const s = data[(4 + c) * N + i];
                    if (s > bestScore) { bestScore = s; bestClass = c; }
                }
            } else {
                const base = i * expectedChannels + 4;
                for (let c = 0; c < numClasses; c++) {
                    const s = data[base + c];
                    if (s > bestScore) { bestScore = s; bestClass = c; }
                }
            }

            if (bestScore > 0) {
                results.push({ x, y, w, h, classId: bestClass, confidence: bestScore });
            }
        }

        results.sort((a, b) => b.confidence - a.confidence);
        return results;
    }

    private applyNMS(
        boxes: Array<{
            x: number; y: number; w: number; h: number;
            classId: number; confidence: number;
        }>
    ): Array<{
        x: number; y: number; w: number; h: number;
        classId: number; confidence: number;
    }> {
        boxes.sort((a, b) => b.confidence - a.confidence);
        const selected: typeof boxes = [];

        while (boxes.length > 0) {
            const current = boxes.shift()!;
            selected.push(current);

            boxes = boxes.filter((box) => {
                // NMS is applied per class only
                if (box.classId !== current.classId) {
                    return true;
                }
                const iou = this.calculateIoU(current, box);
                return iou < this.IOU_THRESHOLD;
            });
        }

        return selected;
    }

    private calculateIoU(
        box1: { x: number; y: number; w: number; h: number },
        box2: { x: number; y: number; w: number; h: number }
    ): number {
        const x1_min = box1.x - box1.w / 2;
        const y1_min = box1.y - box1.h / 2;
        const x1_max = box1.x + box1.w / 2;
        const y1_max = box1.y + box1.h / 2;

        const x2_min = box2.x - box2.w / 2;
        const y2_min = box2.y - box2.h / 2;
        const x2_max = box2.x + box2.w / 2;
        const y2_max = box2.y + box2.h / 2;

        const intersectionXMin = Math.max(x1_min, x2_min);
        const intersectionYMin = Math.max(y1_min, y2_min);
        const intersectionXMax = Math.min(x1_max, x2_max);
        const intersectionYMax = Math.min(y1_max, y2_max);

        const intersectionWidth = Math.max(0, intersectionXMax - intersectionXMin);
        const intersectionHeight = Math.max(0, intersectionYMax - intersectionYMin);
        const intersectionArea = intersectionWidth * intersectionHeight;

        const box1Area = box1.w * box1.h;
        const box2Area = box2.w * box2.h;
        const unionArea = box1Area + box2Area - intersectionArea;

        return unionArea > 0 ? intersectionArea / unionArea : 0;
    }

    pickDetOutput(
        results: Record<string, { data: Float32Array; dims: number[] }>,
        numClasses: number
    ): { data: Float32Array; dims: number[] } {
        const expectedChannels = 4 + numClasses;

        for (const name of Object.keys(results)) {
            const output = results[name];
            const dims = output.dims;

            if (dims.length === 3 && dims[0] === 1) {
                if (dims[1] === expectedChannels || dims[2] === expectedChannels) {
                    if (this.debug) {
                        this.log(`Используем выход '${name}' с размерностью ${dims.join('x')}`);
                    }
                    return output;
                }
            }
        }

        const firstName = Object.keys(results)[0];
        if (this.debug) {
            this.log(`ВНИМАНИЕ: Используем '${firstName}' как выход детекции`);
        }
        return results[firstName];
    }
}