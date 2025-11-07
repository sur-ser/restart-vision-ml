export type ImageSource =
    | string                  // path
    | Buffer                  // raw bytes
    | { data: Buffer; filename?: string; mime?: string }; // extendable variant