import sharp from 'sharp';
import {DetectionAxisAligned} from "../types/detection.types";

export async function cropByDetection(
    imageBuffer: Buffer,
    det: DetectionAxisAligned
): Promise<Buffer> {
    const { x1, y1, x2, y2 } = det.box;
    const left = Math.max(0, Math.min(x1, x2));
    const top  = Math.max(0, Math.min(y1, y2));
    const width  = Math.max(1, Math.abs(x2 - x1));
    const height = Math.max(1, Math.abs(y2 - y1));
    return sharp(imageBuffer).extract({ left, top, width, height }).toBuffer();
}