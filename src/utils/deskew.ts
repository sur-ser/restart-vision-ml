import sharp from 'sharp';
import { openSharp } from './open-sharp';
import type { ImageSource } from '../types/image-source';

/** Fast grayscale + downscale for low-cost rotation */
async function loadSmallGray(src: ImageSource, maxW = 800, maxH = 800) {
    const { sh } = openSharp(src);
    const { data, info } = await sh
        .rotate() // авто-EXIF
        .resize({ width: maxW, height: maxH, fit: 'inside', withoutEnlargement: true })
        .toColourspace('b-w')
        .raw()
        .toBuffer({ resolveWithObject: true });
    // data: Uint8, 1 канал
    return { gray: new Uint8Array(data), width: info.width, height: info.height };
}

/** Otsu thresholding (monochrome 0..255) */
function otsuThreshold(gray: Uint8Array): number {
    const hist = new Uint32Array(256);
    for (let i = 0; i < gray.length; i++) hist[gray[i]]++;

    const total = gray.length;
    let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];

    let sumB = 0, wB = 0, wF = 0, varMax = 0, threshold = 127;
    for (let t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB === 0) continue;
        wF = total - wB;
        if (wF === 0) break;
        sumB += t * hist[t];

        const mB = sumB / wB;
        const mF = (sum - sumB) / wF;
        const between = wB * wF * (mB - mF) * (mB - mF);
        if (between > varMax) {
            varMax = between;
            threshold = t;
        }
    }
    return threshold;
}

/** Rotate a small binary image using Sharp (fast and cheap) */
async function rotateBinary(gray: Uint8Array, w: number, h: number, angleDeg: number) {
    // из массива -> PNG в память -> sharp.rotate -> raw обратно
    const png = await sharp(gray, { raw: { width: w, height: h, channels: 1 } }).png().toBuffer();
    const { data, info } = await sharp(png)
        .rotate(angleDeg, { background: { r: 255, g: 255, b: 255, alpha: 1 } })
        .toColourspace('b-w')
        .raw()
        .toBuffer({ resolveWithObject: true });
    return { bw: new Uint8Array(data), w: info.width, h: info.height };
}

/** Simple metric: how sharp horizontal projections are (variance of row sums) */
function horizontalProjectionScore(bw: Uint8Array, w: number, h: number, thr: number): number {
    let mean = 0;
    const rows = new Float64Array(h);
    for (let y = 0; y < h; y++) {
        let s = 0;
        const rowOff = y * w;
        for (let x = 0; x < w; x++) {
            // бинаризация на лету (черно-белая картинка: 0..255)
            s += (bw[rowOff + x] < thr) ? 1 : 0; // “черный” пиксель
        }
        rows[y] = s;
        mean += s;
    }
    mean /= h;
    let varSum = 0;
    for (let y = 0; y < h; y++) {
        const d = rows[y] - mean;
        varSum += d * d;
    }
    return varSum / h;
}

/** Skew angle detection: coarse then fine search */
export async function estimateSkewAngleSharp(src: ImageSource, opts?: {
    coarseRangeDeg?: number;  // например 8°
    coarseStepDeg?: number;   // напр. 0.5°
    fineStepDeg?: number;     // напр. 0.1°
    maxThumb?: number;        // макс размер миниатюры
}) {
    const coarseRange = opts?.coarseRangeDeg ?? 8;
    const coarseStep  = opts?.coarseStepDeg  ?? 0.5;
    const fineStep    = opts?.fineStepDeg    ?? 0.1;
    const maxThumb    = opts?.maxThumb       ?? 800;

    // 1) Small grayscale image
    const { gray, width: w0, height: h0 } = await loadSmallGray(src, maxThumb, maxThumb);

    // 2) Threshold
    const thr = otsuThreshold(gray);

    // 3) Coarse search
    let bestAngle = 0;
    let bestScore = -Infinity;

    for (let a = -coarseRange; a <= coarseRange + 1e-9; a += coarseStep) {
        const { bw, w, h } = await rotateBinary(gray, w0, h0, a);
        const score = horizontalProjectionScore(bw, w, h, thr);
        if (score > bestScore) { bestScore = score; bestAngle = a; }
    }

    // 4) Fine search around bestAngle
    const fineFrom = bestAngle - coarseStep;
    const fineTo = bestAngle + coarseStep;
    for (let a = fineFrom; a <= fineTo + 1e-9; a += fineStep) {
        const { bw, w, h } = await rotateBinary(gray, w0, h0, a);
        const score = horizontalProjectionScore(bw, w, h, thr);
        if (score > bestScore) { bestScore = score; bestAngle = a; }
    }

    // Normalize confidence to range [0..1] relative to the search window (rough approximation)
    const confidence = Math.max(0, Math.min(1, (bestScore > 0 ? 1 : 0.3)));

    return { angleDeg: bestAngle, score: bestScore, confidence };
}

/** Final deskew — apply the detected rotation angle to the original image (keep size/format) */
export async function deskewWithAngle(src: ImageSource, angleDeg: number): Promise<Buffer> {
    const { sh } = openSharp(src);
    // Один поворот по найденному углу
    return await sh.rotate(angleDeg, { background: { r: 255, g: 255, b: 255, alpha: 1 } }).toBuffer();
}