import {Detection, DetectionAxisAligned} from "../types/detection.types";
import sharp from "sharp";

export async function drawDetections(
    imagePath: string,
    detections: Array<Detection>,
    outDir: string,
    classPalette?: Record<string, string> // optional: { "Receipt": "#22c55e", ... }
): Promise<{ outPath: string; svg: string }> {
    const path = await import('node:path');
    const fs = await import('node:fs/promises');

    // 1) I/O setup
    await fs.mkdir(outDir, { recursive: true });
    const { base } = path.parse(imagePath);
    const outPath = path.join(outDir, base);

    // 2) Image dimensions
    const img = sharp(imagePath);
    const meta = await img.metadata();
    const W = meta.width ?? 0;
    const H = meta.height ?? 0;
    if (!W || !H) throw new Error('Не удалось определить размеры изображения');

    // 3) Color for class (deterministic if not provided in classPalette)
    const colorFor = (label: string) => {
        if (classPalette?.[label]) return classPalette[label];
        // hash → HSL → hex
        let hash = 0;
        for (let i = 0; i < label.length; i++) hash = (hash * 31 + label.charCodeAt(i)) | 0;
        const h = Math.abs(hash) % 360;
        const s = 70; // %
        const l = 50; // %
        // simple HSL→RGB→HEX conversion
        const a = s / 100 * Math.min(l / 100, 1 - l / 100);
        const f = (n: number) => {
            const k = (n + h / 30) % 12;
            const c = l / 100 - a * Math.max(-1, Math.min(k - 3, Math.min(9 - k, 1)));
            return Math.round(255 * c);
        };
        const r = f(0), g = f(8), b = f(4);
        return `#${[r, g, b].map(v => v.toString(16).padStart(2, '0')).join('')}`;
    };

    // 4) Visual parameters
    const strokeWidth = Math.max(2, Math.floor(Math.min(W, H) * 0.003)); // ~0.3% of size
    const fontSize = Math.max(14, Math.floor(Math.min(W, H) * 0.025));   // ~2.5%
    const textPaddingX = Math.max(6, Math.floor(fontSize * 0.5));
    const textPaddingY = Math.max(4, Math.floor(fontSize * 0.35));
    const textColor = '#ffffff';

    // 5) SVG overlay
    //  - draw box
    //  - label background (for readability)
    //  - label text: "Label (0.97)"
    const escape = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g,'&gt;');

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
    <style>
      .lbl { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial; font-size: ${fontSize}px; font-weight: 600; }
    </style>
  `;

    for (const det of detections) {
        const { label, score, box } = det;
        const x1 = Math.max(0, Math.min(W, Math.round(box.x1)));
        const y1 = Math.max(0, Math.min(H, Math.round(box.y1)));
        const x2 = Math.max(0, Math.min(W, Math.round(box.x2)));
        const y2 = Math.max(0, Math.min(H, Math.round(box.y2)));
        if (x2 <= x1 || y2 <= y1) continue;

        const w = x2 - x1;
        const h = y2 - y1;
        const col = colorFor(label);
        const text = `${label} (${(score ?? 0).toFixed(2)})`;
        // Approximate text width (for background sizing): ~0.6 * fontSize * characters
        const approxTextWidth = Math.ceil(text.length * fontSize * 0.6);
        const bgW = approxTextWidth + textPaddingX * 2;
        const bgH = fontSize + textPaddingY * 2;

        // label background coordinates (inside the box, top)
        const bgX = Math.max(0, x1);
        const bgY = Math.max(0, y1 - bgH - Math.max(2, strokeWidth)); // above the border if space allows
        const labelX = bgX + textPaddingX;
        const labelY = bgY + textPaddingY + Math.floor(fontSize * 0.8); // baseline

        svg += `
      <!-- box -->
      <rect x="${x1}" y="${y1}" width="${w}" height="${h}"
            fill="none" stroke="${col}" stroke-width="${strokeWidth}" />

      <!-- label bg -->
      <rect x="${bgX}" y="${Math.max(0, bgY)}"
            width="${Math.min(bgW, W - bgX)}" height="${bgH}"
            fill="${col}" opacity="0.85" rx="${Math.floor(bgH * 0.2)}" />

      <!-- label text -->
      <text x="${labelX}" y="${Math.max(0, labelY)}" class="lbl" fill="${textColor}">
        ${escape(text)}
      </text>
    `;
    }
    svg += `</svg>`;

    // 6) Composite and save
    const out = await sharp(imagePath)
        .composite([{ input: Buffer.from(svg), left: 0, top: 0 }])
        .toFile(outPath);

    return { outPath, svg };
}

export async function drawDetectionsOnBuffer(
    imageBuffer: Buffer,
    detections: DetectionAxisAligned[],
    classPalette?: Record<string, string>
): Promise<Buffer> {
    const img = sharp(imageBuffer);
    const meta = await img.metadata();
    const W = meta.width ?? 0;
    const H = meta.height ?? 0;
    if (!W || !H) throw new Error('Не удалось определить размеры изображения');

    const colorFor = (label: string) => {
        if (classPalette?.[label]) return classPalette[label];
        let hash = 0;
        for (let i = 0; i < label.length; i++) hash = (hash * 31 + label.charCodeAt(i)) | 0;
        const h = Math.abs(hash) % 360, s = 70, l = 50;
        const a = (s / 100) * Math.min(l / 100, 1 - l / 100);
        const f = (n: number) => {
            const k = (n + h / 30) % 12;
            const c = l / 100 - a * Math.max(-1, Math.min(k - 3, Math.min(9 - k, 1)));
            return Math.round(255 * c);
        };
        return `#${[f(0), f(8), f(4)].map(v => v.toString(16).padStart(2, '0')).join('')}`;
    };

    const strokeWidth = Math.max(2, Math.floor(Math.min(W, H) * 0.003));
    const fontSize = Math.max(14, Math.floor(Math.min(W, H) * 0.025));
    const padX = Math.max(6, Math.floor(fontSize * 0.5));
    const padY = Math.max(4, Math.floor(fontSize * 0.35));
    const textColor = '#fff';

    const esc = (s: string) => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
    <style>.lbl{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial;font-size:${fontSize}px;font-weight:600;}</style>`;

    for (const det of detections) {
        const { label, score, box } = det;
        const x1 = Math.max(0, Math.min(W, Math.round(box.x1)));
        const y1 = Math.max(0, Math.min(H, Math.round(box.y1)));
        const x2 = Math.max(0, Math.min(W, Math.round(box.x2)));
        const y2 = Math.max(0, Math.min(H, Math.round(box.y2)));
        if (x2 <= x1 || y2 <= y1) continue;

        const w = x2 - x1, h = y2 - y1, col = colorFor(label);
        const text = `${label} (${(score ?? 0).toFixed(2)})`;
        const approxW = Math.ceil(text.length * fontSize * 0.6);
        const bgW = approxW + padX * 2, bgH = fontSize + padY * 2;
        const bgX = x1, bgY = Math.max(0, y1 - bgH - Math.max(2, strokeWidth));
        const labelX = bgX + padX, labelY = bgY + padY + Math.floor(fontSize * 0.8);

        svg += `
      <rect x="${x1}" y="${y1}" width="${w}" height="${h}" fill="none" stroke="${col}" stroke-width="${strokeWidth}"/>
      <rect x="${bgX}" y="${bgY}" width="${Math.min(bgW, W - bgX)}" height="${bgH}" fill="${col}" opacity="0.85" rx="${Math.floor(bgH*0.2)}"/>
      <text x="${labelX}" y="${labelY}" class="lbl" fill="${textColor}">${esc(text)}</text>
    `;
    }
    svg += `</svg>`;

    return await sharp(imageBuffer).composite([{ input: Buffer.from(svg), left: 0, top: 0 }]).toBuffer();
}