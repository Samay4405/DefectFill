from __future__ import annotations

import asyncio
import json

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

        profile = service.get_elbow_profile()
        if profile is not None:
            await websocket.send_text(json.dumps({"type": "elbow_profile", **profile}))

        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(), timeout=frame_interval_s
                    )
                    command = message.strip().lower()
                    if command in {"ack", "acknowledge", "resume"}:
                        service.acknowledge()
                    continue
                except asyncio.TimeoutError:
                    pass

                if service.is_paused:
                    continue

                out = await service.infer_live()
                packet = build_packet(
                    frame_rgb=out["frame"],
                    heatmap_rgb=out["heatmap_overlay"],
                    score=float(out["score"]),
                    defect=bool(out["defect"]),
                    latency_ms=float(out["latency_ms"]),
                )
                await websocket.send_bytes(packet)

                if bool(out.get("defect", False)):
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "anomaly_detected",
                                "anomaly_id": str(out.get("anomaly_id", "")),
                                "score": float(out.get("score", 0.0)),
                                "threshold": float(out.get("threshold", 0.0)),
                            }
                        )
                    )
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


app = create_app()
