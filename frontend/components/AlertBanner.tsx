"use client";

type Props = {
  show: boolean;
  onAcknowledge: () => void;
};

export function AlertBanner({ show, onAcknowledge }: Props) {
  if (!show) return null;

  return (
    <div className="alert-pulse fixed left-4 top-4 z-50 max-w-xl rounded-2xl border-2 border-[#7a1f12] bg-[#f45b3d] px-5 py-4 text-white shadow-2xl">
      <p className="text-xs uppercase tracking-[0.18em] opacity-90">Critical Quality Alert</p>
      <p className="mt-1 text-lg font-semibold">Surface Defect Detected. Operator Action Required.</p>
      <button
        className="mt-3 rounded-lg bg-[#1f2a28] px-4 py-2 text-sm font-medium text-white hover:bg-[#2a3835]"
        onClick={onAcknowledge}
      >
        Acknowledge
      </button>
    </div>
  );
}
