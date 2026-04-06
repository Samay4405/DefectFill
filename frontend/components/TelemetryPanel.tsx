"use client";

import dynamic from "next/dynamic";

const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

type Point = {
  t: number;
  score: number;
  latencyMs: number;
};

type Props = {
  history: Point[];
  score: number;
  latencyMs: number;
  connected: boolean;
  threshold: number;
  elbowPoints: Array<[number, number]>;
  elbowIndex: number;
};

export function TelemetryPanel({ history, score, latencyMs, connected, threshold, elbowPoints, elbowIndex }: Props) {
  const scoreSeries = history.map((p) => [p.t, p.score]);
  const elbowSeries = elbowPoints.map((p) => [p[0], p[1]]);
  const elbowPoint = elbowPoints[elbowIndex] ?? null;

  const lineOption = {
    animation: false,
    grid: { top: 24, left: 40, right: 16, bottom: 34 },
    xAxis: { type: "time", axisLabel: { color: "#5f706a" }, axisLine: { lineStyle: { color: "#cdd6cd" } } },
    yAxis: {
      type: "value",
      min: 0,
      max: 2,
      axisLabel: { color: "#5f706a" },
      splitLine: { lineStyle: { color: "#dfe6df" } },
    },
    series: [
      {
        type: "line",
        data: scoreSeries,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: "#2f7066" },
        areaStyle: { color: "rgba(47,112,102,0.15)" },
      },
    ],
    tooltip: { trigger: "axis" },
  };

  const gaugeOption = {
    animation: true,
    series: [
      {
        type: "gauge",
        min: 0,
        max: 200,
        progress: { show: true, width: 16, itemStyle: { color: latencyMs > 80 ? "#f45b3d" : "#2f7066" } },
        axisLine: { lineStyle: { width: 16 } },
        pointer: { show: true },
        detail: {
          formatter: `${latencyMs.toFixed(1)} ms`,
          color: "#1f2a28",
          fontSize: 18,
          offsetCenter: [0, "70%"],
        },
        data: [{ value: latencyMs }],
      },
    ],
  };

  const elbowOption = {
    animation: false,
    grid: { top: 24, left: 40, right: 16, bottom: 34 },
    xAxis: { type: "value", min: 0, max: 1, axisLabel: { color: "#5f706a" } },
    yAxis: {
      type: "value",
      axisLabel: { color: "#5f706a" },
      splitLine: { lineStyle: { color: "#dfe6df" } },
    },
    series: [
      {
        type: "line",
        data: elbowSeries,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: "#3a5560" },
      },
      ...(elbowPoint
        ? [
            {
              type: "scatter",
              data: [elbowPoint],
              symbolSize: 10,
              itemStyle: { color: "#f45b3d" },
            },
          ]
        : []),
    ],
    tooltip: { trigger: "axis" },
  };

  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <section className="rounded-2xl border border-[#c9d2c9] bg-[#eef2ee] p-4 shadow-soft">
        <h3 className="text-sm uppercase tracking-[0.15em] text-[#5f706a]">Anomaly Score Trend</h3>
        <div className="mt-2 h-[250px]">
          <ReactECharts option={lineOption} style={{ height: "100%", width: "100%" }} />
        </div>
      </section>

      <section className="rounded-2xl border border-[#c9d2c9] bg-[#eef2ee] p-4 shadow-soft">
        <h3 className="text-sm uppercase tracking-[0.15em] text-[#5f706a]">Elbow Threshold Curve</h3>
        <div className="mt-2 h-[250px]">
          <ReactECharts option={elbowOption} style={{ height: "100%", width: "100%" }} />
        </div>
        <div className="mt-2 flex items-center justify-between">
          <span className="text-sm text-[#5f706a]">Selected Threshold</span>
          <span className="metric text-sm text-[#1f2a28]">{threshold.toFixed(4)}</span>
        </div>
      </section>

      <section className="rounded-2xl border border-[#c9d2c9] bg-[#eef2ee] p-4 shadow-soft">
        <h3 className="text-sm uppercase tracking-[0.15em] text-[#5f706a]">Inference Latency</h3>
        <div className="mt-2 h-[250px]">
          <ReactECharts option={gaugeOption} style={{ height: "100%", width: "100%" }} />
        </div>
        <div className="mt-3 flex items-center justify-between">
          <span className="text-sm text-[#5f706a]">Socket</span>
          <span className={`metric text-sm ${connected ? "text-[#2f7066]" : "text-[#f45b3d]"}`}>
            {connected ? "CONNECTED" : "DISCONNECTED"}
          </span>
        </div>
        <div className="mt-1 flex items-center justify-between">
          <span className="text-sm text-[#5f706a]">Current Score</span>
          <span className="metric text-sm text-[#1f2a28]">{score.toFixed(4)}</span>
        </div>
      </section>
    </div>
  );
}
