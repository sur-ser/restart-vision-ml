import path from "path";
import {openSharp} from "../utils/open-sharp";
import {ImageSource} from "../types/image-source";

interface ImageMetadata {
    width: number;
    height: number;
    format: string;
    colorspace: string;
    hasAlpha: boolean;
    density?: number;
}

export class ImageProcessor {
    async getImageMetadata(src: ImageSource): Promise<ImageMetadata> {
        const meta = await this.asSRGB(src).metadata();
        const width  = meta.width  ?? 0;
        const height = meta.height ?? 0;
        if (width === 0 || height === 0) {
            throw new Error(`Некорректные размеры изображения: ${width}x${height}`);
        }
        return {
            width,
            height,
            format: meta.format || "unknown",
            colorspace: meta.space || "srgb",
            hasAlpha: meta.hasAlpha || false,
            density: meta.density,
        };
    }

    private asSRGB(src: ImageSource) {
        const { sh } = openSharp(src);
        return sh
            .rotate()
            .toColorspace('srgb')
            .removeAlpha();
    }

    async getImageBuffer(src: ImageSource): Promise<Buffer> {
        const { sh } = openSharp(src);
        return await sh.toBuffer();
    }

    async getImageAsBase64(src: ImageSource): Promise<string> {
        const buffer = await this.getImageBuffer(src);
        return buffer.toString("base64");
    }

    async analyzeImageColors(src: ImageSource): Promise<string[]> {
        // Downscale image, convert to sRGB, remove alpha
        const { data, info } = await this.asSRGB(src)
            .resize(64, 64, { fit: 'inside' })
            .raw()
            .toBuffer({ resolveWithObject: true });

        if (info.channels < 3) return [];

        // Quantize into 16 levels per channel (0..15)
        const bins = new Map<string, number>();
        for (let i = 0; i < data.length; i += info.channels) {
            const r = data[i], g = data[i + 1], b = data[i + 2];
            const qr = r >> 4, qg = g >> 4, qb = b >> 4; // 0..15
            const key = `${qr}-${qg}-${qb}`;
            bins.set(key, (bins.get(key) || 0) + 1);
        }

        // Take the top 5 bins and convert to bin centers (x*16 + 8)
        const top = [...bins.entries()].sort((a,b) => b[1]-a[1]).slice(0, 5);
        const colors = top.map(([key]) => {
            const [qr, qg, qb] = key.split('-').map(Number);
            const rr = qr * 16 + 8;
            const gg = qg * 16 + 8;
            const bb = qb * 16 + 8;
            return `rgb(${rr},${gg},${bb})`;
        });

        return colors;
    }

    async detectEdges(src: ImageSource): Promise<number> {
        // 1) Convert to sRGB, grayscale, and downscale for faster processing
        const base = this.asSRGB(src).greyscale().resize(512, 512, { fit: 'inside' });

        // 2) Apply Sobel filters along X and Y
        const sobelX = { width: 3, height: 3, kernel: [-1,0,1,-2,0,2,-1,0,1] };
        const sobelY = { width: 3, height: 3, kernel: [-1,-2,-1, 0,0,0, 1,2,1] };

        const gx = await base.clone().convolve(sobelX).raw().toBuffer({ resolveWithObject: true });
        const gy = await base.clone().convolve(sobelY).raw().toBuffer({ resolveWithObject: true });

        // 3) Average gradient magnitude |G| = sqrt(gx^2 + gy^2)
        const len = gx.data.length; // channels=1
        let sum = 0;
        for (let i = 0; i < len; i++) {
            const x = gx.data[i];
            const y = gy.data[i];
            sum += Math.hypot(x, y);
        }
        // Normalize to [0..1] relative to the max possible (≈1020 for 8‑bit kernels)
        const meanGrad = sum / len;
        const normalized = Math.min(meanGrad / 1020, 1);
        return normalized; // 0..1 — higher is sharper/edgier
    }

    isValidImageFile(filename: string): boolean {
        const validExtensions = [".jpg", ".jpeg", ".png", ".webp", ".tiff"];
        const ext = path.extname(filename).toLowerCase();
        return validExtensions.includes(ext);
    }
}