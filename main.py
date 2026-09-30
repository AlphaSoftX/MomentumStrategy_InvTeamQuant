"""CLI entry point for the production momentum system."""
from __future__ import annotations
import argparse
from pathlib import Path

import config
from pipeline import run
from report import save_result

def main():
    p = argparse.ArgumentParser(description="Run the leakage-safe Nifty momentum research system.")
    p.add_argument("--index", default="nifty500")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--history-start", default=config.HISTORY_START)
    p.add_argument("--history-end", default=config.HISTORY_END)
    p.add_argument("--holdout-start", default=config.FINAL_HOLDOUT_START)
    args = p.parse_args()

    result = run(
        index_name=args.index,
        data_dir=Path(args.data_dir),
        history_start=args.history_start,
        history_end=args.history_end,
        holdout_start=args.holdout_start,
    )

    def show(title, metrics):
        print(f"\n{title}")
        for key, value in metrics.items():
            print(f"  {key:18s}: {value:.6f}" if isinstance(value, float) else f"  {key:18s}: {value}")

    show("Full adaptive walk-forward metrics:", result["metrics"])
    show(f"Adaptive walk-forward slice from {args.holdout_start}:", result["holdout_metrics"])
    show(f"Frozen holdout metrics from {args.holdout_start}:", result["frozen_holdout_metrics"])
    print(f"  frozen alpha mult: {result['frozen_alpha_multiplier']:.6f}")

    print("\nLatest model diagnostics:")
    print(result["ml_diagnostics"].tail(10).to_string(index=False))
    print("\nLatest turnover:")
    print(result["turnover"].tail(10).to_string())

    output_dir = Path(args.data_dir) / config.RESULTS_SUBDIR
    save_result(result, output_dir)
    print(f"\nSaved research artifacts to {output_dir}")

if __name__ == "__main__":
    main()
