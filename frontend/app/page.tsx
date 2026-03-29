"use client";

import { AlertBanner } from "@/components/AlertBanner";
import { InspectionCanvas } from "@/components/InspectionCanvas";
import { TelemetryPanel } from "@/components/TelemetryPanel";
import { useInferenceStream } from "@/hooks/useInferenceStream";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000/ws";

export default function HomePage() {
  const stream = useInferenceStream(WS_URL);

  return (
    <main className="min-h-screen p-4 md:p-8">
      <AlertBanner show={stream.persistentAlert} onAcknowledge={stream.acknowledgeAlert} />

      <header className="mb-6 rounded-2xl border border-[#c9d2c9] bg-[#eef2ee] p-5 shadow-soft">
        <p className="text-xs uppercase tracking-[0.15em] text-[#5f706a]">Industry 4.0 Smart Manufacturing</p>
        <h1 className="mt-1 text-2xl font-semibold text-[#1f2a28] md:text-3xl">Real-Time Quality Control Center</h1>
        <p className="mt-2 max-w-2xl text-sm text-[#5f706a]">
          Calm monitoring mode stays muted during nominal operation. Defect events immediately escalate with persistent
          high-contrast alerting until operator acknowledgement.
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-[1.55fr_1fr]">
        <InspectionCanvas frameBitmap={stream.frameBitmap} heatmapBitmap={stream.heatmapBitmap} />

        <aside className="rounded-2xl border border-[#c9d2c9] bg-[#eef2ee] p-4 shadow-soft">
          <h2 className="text-sm uppercase tracking-[0.15em] text-[#5f706a]">Live Decision</h2>
          <div className="mt-3 grid gap-3">
            <div className="rounded-xl bg-[#f7f9f7] p-3">
              <p className="text-xs text-[#5f706a]">Defect Flag</p>
              <p className={`metric mt-1 text-lg font-semibold ${stream.defect ? "text-[#f45b3d]" : "text-[#2f7066]"}`}>
                {stream.defect ? "DEFECT" : "NOMINAL"}
              </p>
            </div>
            <div className="rounded-xl bg-[#f7f9f7] p-3">
              <p className="text-xs text-[#5f706a]">Anomaly Score</p>
              <p className="metric mt-1 text-lg font-semibold text-[#1f2a28]">{stream.score.toFixed(4)}</p>
            </div>
            <div className="rounded-xl bg-[#f7f9f7] p-3">
              <p className="text-xs text-[#5f706a]">Latency</p>
              <p className={`metric mt-1 text-lg font-semibold ${stream.latencyMs > 80 ? "text-[#f45b3d]" : "text-[#2f7066]"}`}>
                {stream.latencyMs.toFixed(1)} ms
              </p>
            </div>
          </div>
        </aside>
      </section>

      <section className="mt-4">
        <TelemetryPanel
          history={stream.history}
          score={stream.score}
          latencyMs={stream.latencyMs}
          connected={stream.connected}
        />
      </section>
    </main>
  );
}
