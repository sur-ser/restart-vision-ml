export type DetectionMeta = {
    synthetic?: boolean;                // Created by logic, not by the model
    adjusted?: boolean;                 // Geometry adjusted (cropped to container, etc.)
    source?: 'detector'|'refine'|'fallback';
    reason?: string;                    // Brief reason
};

export type DetectionAxisAligned = {
    label: string;
    score: number;
    box: { x1: number; y1: number; x2: number; y2: number };
    meta?: DetectionMeta;
};

export type DetectionOriented = DetectionAxisAligned & {
    angle: number;
};

export type Detection = DetectionAxisAligned | DetectionOriented;

export type AnyDet = DetectionAxisAligned & { meta?: Record<string, any> };