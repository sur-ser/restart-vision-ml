import {ImageSource} from "../types/image-source";
import {DetectionAxisAligned} from "../types/detection.types";

export interface IDetector {
    isReady(): boolean;
    initialize(): Promise<void>;
    detectObjects(src: ImageSource): Promise<{ detections: DetectionAxisAligned[]; width:number; height:number; originalBuffer:Buffer }>;
}