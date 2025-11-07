import {DetectionAxisAligned} from "../types/detection.types";
import {DocumentType} from "../types/document-type.enum";
import {DocumentAnalysisResult} from "../interfaces/document-analysis-result";

/** Configuration and type aliases. */
type Box = { x1:number;y1:number;x2:number;y2:number };
type AnyDet = DetectionAxisAligned & { meta?: Record<string, any> };

type Tol = {
    minBarWidthFrac: number;
    topBottomBandFrac: number;
    maxBarHeightFrac: number;
    insideSlackPx: number;
    classMargin: number;
    minPrimaryAreaFrac: number;

    screenshotMinCoverFrac: number;
    screenshotBarBonus: number;
    screenshotBothBarsBonus: number;
    screenshotCoverBonus: number;
    altCandidateMaxDelta: number;
};

type RefineOptions = {
    tol?: Partial<Tol>;
    allowRetype?: boolean;
    keepDiagnostics?: boolean;
};

const DEFAULT_TOL: Tol = {
    minBarWidthFrac: 0.80,
    topBottomBandFrac: 0.12,
    maxBarHeightFrac: 0.15,
    insideSlackPx: 8,
    classMargin: 0.07,
    minPrimaryAreaFrac: 0.20,

    screenshotMinCoverFrac: 0.80,
    screenshotBarBonus: 0.05,
    screenshotBothBarsBonus: 0.03,
    screenshotCoverBonus: 0.05,
    altCandidateMaxDelta: 0.06,
};

const DEFAULT_OPTS = {
    tol: DEFAULT_TOL,
    allowRetype: true,
    keepDiagnostics: true,
} as const;

type FullRefineConfig = {
    tol: Tol;
    allowRetype: boolean;
    keepDiagnostics: boolean;
};

/** Utilities. */

function clamp(v: number, lo: number, hi: number) {
    return Math.max(lo, Math.min(hi, v));
}
function area(box: Box) {
    return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}
function intersect(a: Box, b: Box): Box {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    return { x1, y1, x2, y2 };
}
function iou(a: Box, b: Box) {
    const inter = intersect(a, b);
    const interA = area(inter);
    const unionA = area(a) + area(b) - interA;
    return unionA > 0 ? interA / unionA : 0;
}
function boxInside(a: Box, b: Box, slackPx: number) {
    return (
        a.x1 >= b.x1 - slackPx &&
        a.y1 >= b.y1 - slackPx &&
        a.x2 <= b.x2 + slackPx &&
        a.y2 <= b.y2 + slackPx
    );
}
function copy<T>(x: T): T {
    return JSON.parse(JSON.stringify(x));
}
function buildConfig(options: RefineOptions = {}): FullRefineConfig {
    const tol: Tol = { ...DEFAULT_TOL, ...(options.tol ?? {}) };
    return {
        tol,
        allowRetype: options.allowRetype ?? DEFAULT_OPTS.allowRetype,
        keepDiagnostics: options.keepDiagnostics ?? DEFAULT_OPTS.keepDiagnostics,
    };
}

/**
 * Validate status/navigation bar without mutating the detection box.
 * - Never change original detection coordinates.
 * - Compute metrics against the container intersection; keep the original box in detections.
 */

function validateBarNoMutate(
    bar: DetectionAxisAligned,
    container: Box,
    W: number,
    H: number,
    which: 'top'|'bottom',
    tol: Tol,
    notes: string[],
): DetectionAxisAligned | null {
    const cW = container.x2 - container.x1;
    const cH = container.y2 - container.y1;
    const minBarW = cW * tol.minBarWidthFrac;
    const maxBarH = cH * tol.maxBarHeightFrac;
    const bandH  = cH * tol.topBottomBandFrac;

    // Use intersection for metrics; do not change the original box.
    const inter = intersect(bar.box as Box, container);
    const interA = area(inter);
    if (interA <= 0) {
        notes.push(`drop ${bar.label}: outside primaryBox`);
        return null;
    }

    const bW = inter.x2 - inter.x1;
    const bH = inter.y2 - inter.y1;
    if (bW < minBarW) {
        notes.push(`drop ${bar.label}: too narrow (${(bW/cW*100).toFixed(1)}% < ${(tol.minBarWidthFrac*100).toFixed(0)}%)`);
        return null;
    }
    if (bH > maxBarH) {
        notes.push(`drop ${bar.label}: too tall (${(bH/cH*100).toFixed(1)}% > ${(tol.maxBarHeightFrac*100).toFixed(0)}%)`);
        return null;
    }

    // Check vertical position using the center of the intersection (without changing the box).
    const cy = (inter.y1 + inter.y2) / 2;
    if (which === 'top') {
        const topBandY2 = container.y1 + bandH;
        if (cy > topBandY2) {
            notes.push(`drop Top status bar: vertical position invalid (cy=${cy.toFixed(1)} > topBand=${topBandY2.toFixed(1)})`);
            return null;
        }
    } else {
        const botBandY1 = container.y2 - bandH;
        if (cy < botBandY1) {
            notes.push(`drop Bottom nav bar: vertical position invalid (cy=${cy.toFixed(1)} < botBand=${botBandY1.toFixed(1)})`);
            return null;
        }
    }

    // Return the original detection (unchanged) and annotate meta.
    const out: AnyDet = copy(bar);
    out.meta = {
        ...(out.meta ?? {}),
        source: 'refine',
        validated: true,
        reason: 'bar_valid',
        // metrics are diagnostic only
        metrics: {
            widthFrac: bW / cW,
            heightFrac: bH / cH,
            which,
        },
    };
    return out;
}

/** Main class selection. */

function pickMainType(
    result: DocumentAnalysisResult,
    tol: Tol,
    notes: string[]
): { type: DocumentType; conf: number; primary?: DetectionAxisAligned } {
    const preferred = new Map<string, DocumentType>([
        ['Receipt', DocumentType.RECEIPT],
        ['Document', DocumentType.DOCUMENT],
        ['Screenshot', DocumentType.SCREENSHOT],
    ]);

    const byClass = new Map<string, DetectionAxisAligned>();
    for (const d of result.detections || []) {
        if (!preferred.has(d.label)) continue;
        const prev = byClass.get(d.label);
        if (!prev || (d.score ?? 0) > (prev.score ?? 0)) byClass.set(d.label, d);
    }

    const cand = Array.from(byClass.entries()).map(([lbl, det]) => ({ lbl, det }));
    if (!cand.length) {
        return { type: result.summary.documentType, conf: result.summary.confidence ?? 0, primary: undefined };
    }

    const hasTop = (result.detections || []).some(d => d.label === 'Top status bar');
    const hasBot = (result.detections || []).some(d => d.label === 'Bottom nav bar');
    const boost = (lbl: string, s: number) =>
        (lbl === 'Screenshot' && (hasTop || hasBot)) ? s + 0.05 : s;

    cand.sort((a, b) => boost(b.lbl, b.det.score ?? 0) - boost(a.lbl, a.det.score ?? 0));

    const best = cand[0];
    const second = cand[1];

    let pick = best;
    if (second && Math.abs((best.det.score ?? 0) - (second.det.score ?? 0)) < tol.classMargin) {
        const ar = result.meta.aspectRatio;
        if (hasTop || hasBot) {
            pick = cand.find(c => c.lbl === 'Screenshot') || pick;
            notes.push('tie-breaker: bars present → prefer Screenshot');
        } else if (ar >= 0.65 && byClass.has('Document')) {
            pick = cand.find(c => c.lbl === 'Document') || pick;
            notes.push('tie-breaker: AR≈A4 & no bars → prefer Document');
        }
    }

    const type = preferred.get(pick.lbl)!;
    return { type, conf: pick.det.score ?? 0, primary: pick.det };
}

/** Compute Screenshot score with bonuses (bars and coverage). */

function scoreScreenshot(
    container: Box,
    topOk: DetectionAxisAligned | null,
    botOk: DetectionAxisAligned | null,
    frameW: number,
    frameH: number,
    baseConf: number,
    tol: Tol
) {
    const cover = area(container) / (frameW * frameH);
    let score = baseConf;
    let validBars = 0;
    if (topOk) { score += tol.screenshotBarBonus; validBars++; }
    if (botOk) { score += tol.screenshotBarBonus; validBars++; }
    if (validBars === 2) score += tol.screenshotBothBarsBonus;
    if (cover >= tol.screenshotMinCoverFrac) score += tol.screenshotCoverBonus;

    return {score, cover, validBars};
}

function upsertPrimaryDetection(
    out: DocumentAnalysisResult,
    notes: string[],
    wantedLbl?: string,
) {
    const pb = out.summary.primaryBox as Box | undefined;
    if (!pb || !wantedLbl) return;
    out.detections = out.detections || [];

    let bestIou = 0;
    for (let i = 0; i < out.detections.length; i++) {
        const d = out.detections[i];
        if (d.label !== wantedLbl) continue;
        const v = iou(d.box as Box, pb);
        if (v > bestIou) bestIou = v;
    }
    if (bestIou >= 0.90) return;

    const det: AnyDet = {
        label: wantedLbl,
        score: out.summary.confidence ?? 0,
        box: { ...pb },
        meta: { synthetic: true, source: 'refine', reason: 'matches_primaryBox' },
    };
    out.detections.push(det);
    notes.push('detections: inserted synthetic main-class box to match primaryBox');
}

/** Main refinement pipeline. */

export function refineDocumentAnalysis(
    input: DocumentAnalysisResult,
    options?: RefineOptions
): DocumentAnalysisResult {
    const cfg = buildConfig(options);
    const notes: string[] = [];
    const W = input.meta.width;
    const H = input.meta.height;

    // 1) Pick a base class
    let { type: mainType, conf: mainConf, primary } = pickMainType(input, cfg.tol, notes);

    // 2) Retype if allowed
    let summary = copy(input.summary);
    if (cfg.allowRetype && mainType !== input.summary.documentType) {
        notes.push(`retype: ${input.summary.documentType} → ${mainType}`);
        summary.documentType = mainType;
        summary.confidence = mainConf;
    }

    // 3) Primary box
    if (!summary.primaryBox && primary) {
        summary.primaryBox = copy(primary.box);
        notes.push('primaryBox: adopted from best main-class detection');
    }
    if (summary.primaryBox) {
        summary.primaryBox.x1 = clamp(summary.primaryBox.x1, 0, W);
        summary.primaryBox.x2 = clamp(summary.primaryBox.x2, 0, W);
        summary.primaryBox.y1 = clamp(summary.primaryBox.y1, 0, H);
        summary.primaryBox.y2 = clamp(summary.primaryBox.y2, 0, H);

        const primA = area(summary.primaryBox as Box);
        const frameA = W * H;
        const cover = frameA > 0 ? (primA / frameA) : 0;
        const wasPartial = !!summary.quality?.isPartial;
        const isPartial = cover < cfg.tol.minPrimaryAreaFrac;
        if (isPartial !== wasPartial) {
            summary.quality = summary.quality || {} as any;
            summary.quality.isPartial = isPartial;
            if (isPartial) notes.push(`flag partial: primary area ${(cover*100).toFixed(1)}% < ${(cfg.tol.minPrimaryAreaFrac*100).toFixed(0)}%`);
        }
    }

    // 4) Extract bars
    let outDetections: DetectionAxisAligned[] = [];
    const bars: DetectionAxisAligned[] = [];
    const rest: DetectionAxisAligned[] = [];
    for (const d of input.detections || []) {
        if (d.label === 'Top status bar' || d.label === 'Bottom nav bar') bars.push(d);
        else rest.push(d);
    }
    outDetections = rest;

    // 5) If not a Screenshot — drop bars
    if (summary.documentType !== DocumentType.SCREENSHOT && bars.length) {
        notes.push('drop bars: non-screenshot type');
    }

    // 6) Screenshot logic
    if (summary.documentType === DocumentType.SCREENSHOT) {
        const bestScreenshot = (input.detections || [])
            .filter(d => d.label === 'Screenshot')
            .sort((a,b) => (b.score??0) - (a.score??0))[0];

        // Screenshot container: best detection or full frame
        let screenshotContainer: Box = bestScreenshot
            ? copy(bestScreenshot.box)
            : { x1: 0, y1: 0, x2: W, y2: H };

        // Validate bars without changing coordinates
        const topRaw = bars.find(b => b.label === 'Top status bar') || null;
        const botRaw = bars.find(b => b.label === 'Bottom nav bar') || null;

        const topOk = topRaw ? validateBarNoMutate(topRaw, screenshotContainer, W, H, 'top', cfg.tol, notes) : null;
        const botOk = botRaw ? validateBarNoMutate(botRaw, screenshotContainer, W, H, 'bottom', cfg.tol, notes) : null;

        const baseConf = summary.confidence ?? (bestScreenshot?.score ?? 0);
        const s = scoreScreenshot(screenshotContainer, topOk, botOk, W, H, baseConf, cfg.tol);

        // Alternative (Document/Receipt) candidate
        const alt = (input.detections || [])
            .filter(d => d.label === 'Document' || d.label === 'Receipt')
            .sort((a,b) => (b.score??0) - (a.score??0))[0];

        let altScore = -Infinity;
        let altType: DocumentType | undefined;
        if (alt) {
            altScore = (alt.score ?? 0) + 0.5 * iou(alt.box as Box, screenshotContainer);
            altType = (alt.label === 'Document') ? DocumentType.DOCUMENT : DocumentType.RECEIPT;
        }

        const needSwitchToAlt =
            (s.cover < cfg.tol.screenshotMinCoverFrac) &&
            (altScore > -Infinity) &&
            (altScore >= s.score - cfg.tol.altCandidateMaxDelta);

        if (needSwitchToAlt && altType && cfg.allowRetype) {
            notes.push(
                `switch: Screenshot → ${altType} (cover ${(s.cover*100).toFixed(1)}% < ${(cfg.tol.screenshotMinCoverFrac*100).toFixed(0)}%, alt≈${altScore.toFixed(3)} vs scr≈${s.score.toFixed(3)})`
            );
            summary.documentType = altType;
            summary.confidence = alt.score ?? summary.confidence ?? 0;
            summary.primaryBox = copy(alt.box);
            // Do not add bars
        } else {
            // Keep Screenshot
            const containerCover = area(screenshotContainer) / (W * H);
            if (!bestScreenshot || containerCover < cfg.tol.screenshotMinCoverFrac) {
                summary.primaryBox = { x1: 0, y1: 0, x2: W, y2: H };
                notes.push('primaryBox: forced to full frame for Screenshot');
            } else {
                summary.primaryBox = copy(screenshotContainer);
            }
            summary.confidence = s.score;

            // Append valid bars (original boxes, unchanged)
            if (topOk) outDetections.push(topOk);
            if (botOk) outDetections.push(botOk);
        }
    }

    // 7) Resolve Document vs Receipt
    const doc = outDetections.filter(d => d.label === 'Document');
    const rec = outDetections.filter(d => d.label === 'Receipt');
    if (doc.length && rec.length && summary.primaryBox && cfg.allowRetype) {
        const scoreDoc = Math.max(...doc.map(d => d.score ?? 0));
        const scoreRec = Math.max(...rec.map(d => d.score ?? 0));
        const iouDoc = Math.max(...doc.map(d => iou(d.box as Box, summary.primaryBox! as Box)));
        const iouRec = Math.max(...rec.map(d => iou(d.box as Box, summary.primaryBox! as Box)));

        const keep = (scoreDoc + iouDoc) >= (scoreRec + iouRec) ? 'Document' : 'Receipt';
        outDetections = outDetections.filter(d => (d.label === keep) || (d.label !== 'Document' && d.label !== 'Receipt'));

        const wantType = (keep === 'Document') ? DocumentType.DOCUMENT : DocumentType.RECEIPT;
        if (summary.documentType !== wantType) {
            summary.documentType = wantType;
            summary.confidence = (keep === 'Document') ? scoreDoc : scoreRec;
            notes.push(`resolve doc/receipt conflict → prefer ${keep}`);
        }
    }

    // 8) Replace primary with the best class box if IoU is low
    if (cfg.allowRetype && summary.primaryBox) {
        const wantedLbl =
            summary.documentType === DocumentType.SCREENSHOT ? 'Screenshot' :
                summary.documentType === DocumentType.DOCUMENT  ? 'Document'   :
                    summary.documentType === DocumentType.RECEIPT   ? 'Receipt'    : undefined;

        const cands = outDetections.filter(d => d.label === (wantedLbl || ''));
        if (cands.length) {
            const scored = cands
                .map(d => ({ d, score: (d.score ?? 0) + 0.5 * iou(d.box as Box, summary.primaryBox! as Box) }))
                .sort((a, b) => b.score - a.score);
            const best = scored[0].d;
            if (iou(best.box as Box, summary.primaryBox as Box) < 0.5) {
                summary.primaryBox = copy(best.box);
                notes.push(`primaryBox: replaced by best ${wantedLbl} (low IoU with old)`);
            }
        }
    }

    // 9) Fallback for Unknown — synthetic full-frame
    if (summary.documentType === DocumentType.UNKNOWN) {
        const fullDoc: AnyDet = {
            label: 'Document',
            score: 0,
            box: { x1: 0, y1: 0, x2: W, y2: H },
            meta: { synthetic: true, source: 'fallback', reason: 'unknown_full_frame' },
        };

        if (!summary.primaryBox) {
            summary.primaryBox = { ...fullDoc.box };
            notes.push('primaryBox: synthesized full-frame for Unknown');
        }

        const hasNearFullDoc = (input.detections ?? []).some(d =>
            (d.label === 'Document' || d.label === 'Screenshot') &&
            iou(d.box as Box, fullDoc.box) > 0.95
        );
        if (!hasNearFullDoc) {
            outDetections = [fullDoc, ...outDetections];
            notes.push('synth detection: Document full-frame for Unknown');
        }

        summary.quality = summary.quality ?? {} as any;
        summary.quality.isPartial = false;
    }

    // 10) Assemble result
    const out: DocumentAnalysisResult = {
        ...input,
        summary,
        detections: outDetections,
    };

    // 11) Invariant: ensure primary is reflected in detections (synthetic if needed)
    const wantedLbl =
        out.summary.documentType === DocumentType.SCREENSHOT ? 'Screenshot' :
            out.summary.documentType === DocumentType.DOCUMENT  ? 'Document'   :
                out.summary.documentType === DocumentType.RECEIPT   ? 'Receipt'    : undefined;
    upsertPrimaryDetection(out, notes, wantedLbl);

    if (cfg.keepDiagnostics) {
        out.signals = {
            ...(out.signals || {}),
            diagnostics: [...(out.signals as any)?.diagnostics ?? [], ...notes],
        } as any;
    }
    return out;
}