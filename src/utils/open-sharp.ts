import sharp, { Sharp } from 'sharp';
import path from 'node:path';
import type { ImageSource } from '../types/image-source';

export function openSharp(src: ImageSource): { sh: Sharp; filename?: string } {
    if (typeof src === 'string') {
        return { sh: sharp(src), filename: path.basename(src) };
    }
    if (Buffer.isBuffer(src)) {
        return { sh: sharp(src) };
    }
    // { data: Buffer, ... }
    return { sh: sharp(src.data), filename: src.filename };
}