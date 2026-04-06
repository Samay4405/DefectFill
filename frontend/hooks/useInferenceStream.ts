"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { parseInferencePacket } from "@/lib/protocol";

export type StreamPoint = {
  t: number;
  score: number;
  latencyMs: number;
};

export type ElbowProfile = {
  threshold: number;
  elbowIndex: number;
  points: Array<[number, number]>;
};

export type StreamState = {
  connected: boolean;
  frameBitmap: ImageBitmap | null;
  heatmapBitmap: ImageBitmap | null;
  score: number;
  latencyMs: number;
  defect: boolean;
  threshold: number;
  latestAnomalyId: string;
  history: StreamPoint[];
  elbowProfile: ElbowProfile | null;
  persistentAlert: boolean;
  acknowledgeAlert: () => void;
};

async function decodeImage(bytes: Uint8Array): Promise<ImageBitmap> {
  const blob = new Blob([bytes]);
  return await createImageBitmap(blob);
}

export function useInferenceStream(wsUrl: string): StreamState {
  const [connected, setConnected] = useState(false);
  const [frameBitmap, setFrameBitmap] = useState<ImageBitmap | null>(null);
  const [heatmapBitmap, setHeatmapBitmap] = useState<ImageBitmap | null>(null);
  const [score, setScore] = useState(0);
  const [latencyMs, setLatencyMs] = useState(0);
  const [defect, setDefect] = useState(false);
  const [threshold, setThreshold] = useState(0);
  const [latestAnomalyId, setLatestAnomalyId] = useState("");
  const [history, setHistory] = useState<StreamPoint[]>([]);
  const [elbowProfile, setElbowProfile] = useState<ElbowProfile | null>(null);
  const [persistentAlert, setPersistentAlert] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const rafRef = useRef<number | null>(null);

  const acknowledgeAlert = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send("acknowledge");
    }
    setPersistentAlert(false);
  }, []);

  useEffect(() => {
    let active = true;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      if (!active) return;
      setConnected(true);
      ws.send("subscribe");
    };

    ws.onmessage = async (event: MessageEvent<ArrayBuffer | string>) => {
      if (!active) {
        return;
      }

      if (typeof event.data === "string") {
        try {
          const data = JSON.parse(event.data) as {
            type?: string;
            threshold?: number;
            elbow_index?: number;
            points?: Array<[number, number]>;
            anomaly_id?: string;
            score?: number;
          };

          if (data.type === "elbow_profile") {
            setElbowProfile({
              threshold: Number(data.threshold ?? 0),
              elbowIndex: Number(data.elbow_index ?? 0),
              points: Array.isArray(data.points) ? data.points : [],
            });
            setThreshold(Number(data.threshold ?? 0));
          }

          if (data.type === "anomaly_detected") {
            setPersistentAlert(true);
            setLatestAnomalyId(String(data.anomaly_id ?? ""));
            if (typeof data.threshold === "number") {
              setThreshold(data.threshold);
            }
            if (typeof data.score === "number") {
              setScore(data.score);
            }
          }
        } catch {
          // Ignore malformed control message.
        }
        return;
      }

      try {
        const packet = parseInferencePacket(event.data);
        const [frame, heat] = await Promise.all([
          decodeImage(packet.frameBytes),
          decodeImage(packet.heatmapBytes),
        ]);

        if (!active) {
          frame.close();
          heat.close();
          return;
        }

        setFrameBitmap((prev: ImageBitmap | null) => {
          prev?.close();
          return frame;
        });
        setHeatmapBitmap((prev: ImageBitmap | null) => {
          prev?.close();
          return heat;
        });

        setScore(packet.score);
        setLatencyMs(packet.latencyMs);
        setDefect(packet.defect);
        if (packet.defect) {
          setPersistentAlert(true);
        }

        const now = performance.now();
        setHistory((prev: StreamPoint[]) => {
          const next = [...prev, { t: now, score: packet.score, latencyMs: packet.latencyMs }];
          return next.slice(-120);
        });
      } catch {
        // Ignore malformed packet and continue.
      }
    };

    ws.onclose = () => {
      if (!active) return;
      setConnected(false);
    };

    ws.onerror = () => {
      if (!active) return;
      setConnected(false);
    };

    wsRef.current = ws;

    return () => {
      active = false;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
      }
      ws.close();
    };
  }, [wsUrl]);

  return useMemo(
    () => ({
      connected,
      frameBitmap,
      heatmapBitmap,
      score,
      latencyMs,
      defect,
      threshold,
      latestAnomalyId,
      history,
      elbowProfile,
      persistentAlert,
      acknowledgeAlert,
    }),
    [
      connected,
      frameBitmap,
      heatmapBitmap,
      score,
      latencyMs,
      defect,
      threshold,
      latestAnomalyId,
      history,
      elbowProfile,
      persistentAlert,
      acknowledgeAlert,
    ]
  );
}
