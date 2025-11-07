import path from "node:path";
import { promises as fs } from "node:fs";
import sharp from "sharp";
import {deskewWithAngle, estimateSkewAngleSharp} from "../src/utils/deskew";
import {openSharp} from "../src/utils/open-sharp";


const [, , imagePathArg] = process.argv;
if (!imagePathArg) {
    console.error("Usage: tsx examples/04-skew-and-deskew.ts path/to/image.jpg");
    process.exit(1);
}
const imagePath = path.resolve(process.cwd(), imagePathArg);
const outDir = path.resolve(process.cwd(), "images/processed");
await fs.mkdir(outDir, { recursive: true });

const { angleDeg, confidence } = await estimateSkewAngleSharp(imagePath, {
    coarseRangeDeg: 8,
    coarseStepDeg: 0.5,
    fineStepDeg: 0.1,
    maxThumb: 800,
});

console.log(`Skew angle ~ ${angleDeg.toFixed(2)}°, confidence=${confidence.toFixed(2)}`);

const DO_SKEW = confidence >= 0.5 && Math.abs(angleDeg) >= 0.3;
const outBuf = DO_SKEW ? await deskewWithAngle(imagePath, angleDeg) : (await openSharp(imagePath).sh.toBuffer());

const outPath = path.join(outDir, `deskewed-${path.basename(imagePath)}`);
await sharp(outBuf).toFile(outPath);
console.log("Saved:", outPath);