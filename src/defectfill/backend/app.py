from __future__ import annotations

import asyncio
from contextlib import suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from defectfill.backend.protocol import build_packet
from defectfill.backend.stream_service import StreamInferenceService
from defectfill.config import load_config


def create_app(config_path: str = "configs/default.yaml") -> FastAPI:
    cfg = load_config(config_path)
    stream_cfg = cfg.setdefault("stream", {})
    stream_cfg.setdefault("target_fps", 12)

    app = FastAPI(title="DefectFill Smart Manufacturing Backend", version="0.2.0")
    service: StreamInferenceService | None = None

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "target_fps": stream_cfg["target_fps"],
            "category": cfg["dataset"]["category"],
            "max_latency_ms": cfg["inference"].get("max_latency_ms", 80),
        }

    @app.websocket("/ws")
    async def ws_inference(websocket: WebSocket):
        nonlocal service
        await websocket.accept()
        frame_interval_s = 1.0 / max(1, int(stream_cfg["target_fps"]))

        if service is None:
            service = StreamInferenceService(cfg)

        send_task = None

        async def loop_send():
            while True:
                out = await service.infer_live()
                packet = build_packet(
                    frame_rgb=out["frame"],
                    heatmap_rgb=out["heatmap_overlay"],
                    score=float(out["score"]),
                    defect=bool(out["defect"]),
                    latency_ms=float(out["latency_ms"]),
                )
                await websocket.send_bytes(packet)
                await asyncio.sleep(frame_interval_s)

        try:
            send_task = asyncio.create_task(loop_send())
            while True:
                # Keep socket alive and allow operator control messages in future.
                _ = await websocket.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if send_task is not None:
                send_task.cancel()
                with suppress(asyncio.CancelledError):
                    await send_task

    return app


app = create_app()
