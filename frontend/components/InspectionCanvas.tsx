"use client";

import { useEffect, useRef } from "react";

type Props = {
  frameBitmap: ImageBitmap | null;
  heatmapBitmap: ImageBitmap | null;
};

export function InspectionCanvas({ frameBitmap, heatmapBitmap }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current || !frameBitmap) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = frameBitmap.width;
    canvas.height = frameBitmap.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(frameBitmap, 0, 0, canvas.width, canvas.height);

    if (heatmapBitmap) {
      ctx.globalAlpha = 0.56;
      ctx.drawImage(heatmapBitmap, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
    }
  }, [frameBitmap, heatmapBitmap]);

  return (
    <div className="rounded-2xl border border-[#c9d2c9] bg-[#edf1ed] p-3 shadow-soft">
      <canvas ref={canvasRef} className="h-auto w-full rounded-xl" />
    </div>
  );
}
