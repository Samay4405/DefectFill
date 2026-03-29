export const MAGIC = "DFWS";
export const HEADER_SIZE = 36;

export type InferencePacket = {
  width: number;
  height: number;
  score: number;
  latencyMs: number;
  defect: boolean;
  timestampNs: bigint;
  frameBytes: Uint8Array;
  heatmapBytes: Uint8Array;
};

export function parseInferencePacket(buffer: ArrayBuffer): InferencePacket {
  const view = new DataView(buffer);

  const m0 = String.fromCharCode(view.getUint8(0));
  const m1 = String.fromCharCode(view.getUint8(1));
  const m2 = String.fromCharCode(view.getUint8(2));
  const m3 = String.fromCharCode(view.getUint8(3));
  const magic = `${m0}${m1}${m2}${m3}`;
  if (magic !== MAGIC) {
    throw new Error(`Invalid packet magic: ${magic}`);
  }

  const version = view.getUint16(4, true);
  if (version !== 1) {
    throw new Error(`Unsupported protocol version: ${version}`);
  }

  const flags = view.getUint16(6, true);
  const width = view.getUint16(8, true);
  const height = view.getUint16(10, true);
  const frameLen = view.getUint32(12, true);
  const heatLen = view.getUint32(16, true);
  const score = view.getFloat32(20, true);
  const latencyMs = view.getFloat32(24, true);
  const timestampNs = view.getBigUint64(28, true);

  const payload = new Uint8Array(buffer, HEADER_SIZE);
  const frameBytes = payload.slice(0, frameLen);
  const heatmapBytes = payload.slice(frameLen, frameLen + heatLen);

  return {
    width,
    height,
    score,
    latencyMs,
    defect: (flags & 1) === 1,
    timestampNs,
    frameBytes,
    heatmapBytes,
  };
}
