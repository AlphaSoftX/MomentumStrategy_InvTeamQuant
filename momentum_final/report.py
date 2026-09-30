"""Persist pipeline outputs as portable CSV/JSON research artifacts."""
from __future__ import annotations
import json
from pathlib import Path

def save_result(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result["metrics"], default=float, indent=2))
    (output / "equity_curve.csv").write_text(result["equity_curve"].to_csv())
    result["turnover"].to_csv(output / "turnover.csv")
    result["ml_diagnostics"].to_csv(output / "ml_diagnostics.csv", index=False)
    if "frozen_holdout_equity" in result:
        result["frozen_holdout_equity"].to_csv(output / "frozen_holdout_equity.csv")
        result["frozen_holdout_turnover"].to_csv(output / "frozen_holdout_turnover.csv")
        (output / "frozen_holdout_metrics.json").write_text(
            json.dumps(result["frozen_holdout_metrics"], default=float, indent=2)
        )
        (output / "frozen_alpha.json").write_text(
            json.dumps({"alpha_multiplier": result["frozen_alpha_multiplier"]}, indent=2)
        )
    return output
