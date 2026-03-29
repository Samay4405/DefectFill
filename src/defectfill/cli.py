from __future__ import annotations

import argparse
import json

from .config import load_config
from .pipeline import DefectFillPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DefectFill: TensorFlow industrial anomaly pipeline")
    p.add_argument("--config", type=str, default="configs/default.yaml")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("synthesize", help="Run phase-1 synthetic defect generation")
    sub.add_parser("build-memory", help="Build PatchCore memory bank from nominal train images")

    infer = sub.add_parser("infer", help="Run anomaly inference on test split")
    infer.add_argument("--defect-type", type=str, default="good")

    return p


def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    pipe = DefectFillPipeline(cfg)

    if args.command == "synthesize":
        pipe.run_phase1_synthesis()
        print("Synthetic generation complete.")
        return

    if args.command == "build-memory":
        artifacts = pipe.build_memory_bank()
        print(json.dumps(artifacts.__dict__, indent=2))
        return

    if args.command == "infer":
        scores = pipe.infer_folder(defect_type=args.defect_type)
        print(json.dumps(scores, indent=2))
        return


if __name__ == "__main__":
    main()
