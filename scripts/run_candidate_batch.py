from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_project_root

ROOT = ensure_project_root()

from src.evaluation import (  # noqa: E402
    BASELINE_EXPERIMENTS,
    EXPERIMENTS,
    CandidateBatchRunner,
    DatasetSpec,
    finalize_decisions,
    results_to_frame,
    build_dataset_bundle,
)
from src.evaluation.registry import ExperimentSpec  # noqa: E402


DEFAULT_REPORT_PATH = ROOT / "reports" / "candidate_offline_selection_2026-04-14.md"
DEFAULT_RUN_ROOT = ROOT / "artifacts" / "experiments" / "runs"
DEFAULT_DATASET_ROOT = ROOT / "artifacts" / "experiments" / "datasets"
DECISION_PRIORITY = {
    "reference": 0,
    "promote": 1,
    "keep": 2,
    "archive": 3,
    "pending": 4,
}


def _parse_symbols(value: str) -> tuple[str, ...]:
    symbols = [item.strip() for item in value.split(",") if item.strip()]
    if not symbols:
        raise argparse.ArgumentTypeError("At least one symbol is required")
    return tuple(symbols)


def _resolve_dates(args: argparse.Namespace) -> tuple[str, str]:
    if args.start_date and args.end_date:
        return args.start_date, args.end_date

    end_date = args.end_date
    if end_date is None:
        end_date = datetime.now(timezone.utc).date().isoformat()
    if args.start_date is not None:
        return args.start_date, end_date

    start_date = (pd.Timestamp(end_date) - timedelta(days=args.days)).date().isoformat()
    return start_date, end_date


def _select_specs(suite: str, candidate_ids: list[str] | None) -> list[ExperimentSpec]:
    if suite == "baselines":
        specs = list(BASELINE_EXPERIMENTS)
    elif suite == "shortlist":
        specs = list(EXPERIMENTS)
    elif suite == "full":
        specs = list(EXPERIMENTS)
    else:
        raise ValueError(f"Unknown suite: {suite}")

    if candidate_ids:
        wanted = {candidate_id.upper() for candidate_id in candidate_ids}
        specs = [spec for spec in specs if spec.pipeline_id.upper() in wanted]

    return specs


def _fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def _render_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    if frame.empty:
        return "_No rows_"

    display = frame.copy()
    lines = ["| " + " | ".join(header for header, _ in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in display.iterrows():
        values = [_fmt(row.get(column, "")) if isinstance(row.get(column, ""), (float, int)) else str(row.get(column, "")) for _, column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _ranked_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = frame.copy()
    summary["_decision_rank"] = summary["decision"].map(DECISION_PRIORITY).fillna(99).astype(int)
    summary = summary.sort_values(
        ["_decision_rank", "test_sharpe_ratio", "test_total_return_pct"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return summary.drop(columns=["_decision_rank"])


def _render_report(
    bundle,
    result,
    results_frame: pd.DataFrame,
    *,
    suite: str,
    selected_ids: list[str] | None,
) -> str:
    baseline = result.baseline_outcome()
    summary = results_frame.copy()
    summary["test_total_return_pct"] = summary["test_summary.total_return_pct"]
    summary["test_sharpe_ratio"] = summary["test_summary.sharpe_ratio"]
    summary["test_max_drawdown_pct"] = summary["test_summary.max_drawdown_pct"]
    summary["validation_total_return_pct"] = summary["validation_summary.total_return_pct"]
    summary["validation_sharpe_ratio"] = summary["validation_summary.sharpe_ratio"]
    summary["validation_max_drawdown_pct"] = summary["validation_summary.max_drawdown_pct"]
    summary = _ranked_summary(summary)

    promoted = int((summary["decision"] == "promote").sum())
    kept = int((summary["decision"] == "keep").sum())
    archived = int((summary["decision"] == "archive").sum())
    references = int((summary["decision"] == "reference").sum())
    direct = int((summary["implementation_mode"] == "direct").sum())
    proxy = int((summary["implementation_mode"] == "proxy").sum())

    paper_shadow = summary[summary["decision"] == "promote"].head(5)
    top = summary.head(8)

    lines = [
        "# Offline Candidate Selection Report",
        "",
        f"- generated_at: `{result.created_at}`",
        f"- suite: `{suite}`",
        f"- dataset_version: `{bundle.version}`",
        f"- dataset_root: `{bundle.root}`",
        f"- symbols: `{', '.join(bundle.spec.symbols)}`",
        f"- split: train={len(result.split.train):,}, validation={len(result.split.validation):,}, test={len(result.split.test):,}",
        f"- selected_ids: `{', '.join(selected_ids) if selected_ids else 'all'}`",
        "",
        "## Protocol",
        "",
        "- Single fixed dataset snapshot with a train/validation/test expanding split.",
        "- Costs are applied with the same long-only rails as the rest of the repo.",
        "- `F001` is the canonical reference baseline for the candidate sweep.",
        "- Price-only proxy features are used for doc families that still lack dedicated fundamental/news feeds.",
        "",
        "## Suite Summary",
        "",
        f"- reference_runs: {references}",
        f"- promoted: {promoted}",
        f"- kept: {kept}",
        f"- archived: {archived}",
        f"- direct_modes: {direct}",
        f"- proxy_modes: {proxy}",
        "",
        "## Baseline Reference",
        "",
        _render_table(
            summary[summary["pipeline_id"] == baseline.spec.pipeline_id],
            [
                ("Pipeline", "pipeline_id"),
                ("Mode", "implementation_mode"),
                ("Base Model", "base_model"),
                ("Return %", "test_total_return_pct"),
                ("Sharpe", "test_sharpe_ratio"),
                ("Max DD %", "test_max_drawdown_pct"),
                ("Decision", "decision"),
            ],
        ),
        "",
        "## Candidate Sweep",
        "",
        _render_table(
            summary[summary["pipeline_id"] != baseline.spec.pipeline_id],
            [
                ("Pipeline", "pipeline_id"),
                ("Family", "family"),
                ("Mode", "implementation_mode"),
                ("Base Model", "base_model"),
                ("Features", "feature_profile"),
                ("Overlays", "overlays"),
                ("Test Return %", "test_total_return_pct"),
                ("Test Sharpe", "test_sharpe_ratio"),
                ("Test Max DD %", "test_max_drawdown_pct"),
                ("Decision", "decision"),
            ],
        ),
        "",
        "## Top Candidates",
        "",
        _render_table(
            top,
            [
                ("Pipeline", "pipeline_id"),
                ("Family", "family"),
                ("Mode", "implementation_mode"),
                ("Test Return %", "test_total_return_pct"),
                ("Test Sharpe", "test_sharpe_ratio"),
                ("Test Max DD %", "test_max_drawdown_pct"),
                ("Decision", "decision"),
            ],
        ),
        "",
        "## Paper Shadow Shortlist",
        "",
        _render_table(
            paper_shadow,
            [
                ("Pipeline", "pipeline_id"),
                ("Family", "family"),
                ("Mode", "implementation_mode"),
                ("Test Return %", "test_total_return_pct"),
                ("Test Sharpe", "test_sharpe_ratio"),
                ("Test Max DD %", "test_max_drawdown_pct"),
                ("Decision", "decision"),
            ],
        ),
        "",
        "## Notes",
        "",
        "- `B011`, `B018`, and `F025` are executed as price-only proxies until dedicated fundamental feeds are wired in.",
        "- `F031` is included as an additional operational overlay baseline for comparison.",
        "- If a candidate beats the baseline only on raw return but not on Sharpe or drawdown, it is kept rather than promoted.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the shortlist candidate batch on a fixed dataset snapshot.")
    parser.add_argument("--suite", choices=["baselines", "shortlist", "full"], default="shortlist")
    parser.add_argument("--candidate-ids", nargs="*", help="Optional subset of pipeline IDs to execute.")
    parser.add_argument("--symbols", type=_parse_symbols, default=_parse_symbols(",".join(["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD"])))
    parser.add_argument("--days", type=int, default=180, help="Lookback window if explicit start/end dates are omitted.")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--refresh-dataset", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--baseline-pipeline-id", type=str, default="F001")
    args = parser.parse_args()

    start_date, end_date = _resolve_dates(args)
    dataset_spec = DatasetSpec(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        source="alpaca_or_synthetic",
        target_horizon=1,
    )

    bundle = build_dataset_bundle(
        dataset_spec,
        cache_root=args.dataset_root,
        refresh=args.refresh_dataset,
    )

    specs = _select_specs(args.suite, args.candidate_ids)
    if not specs:
        raise SystemExit("No experiments selected")
    if args.baseline_pipeline_id not in {spec.pipeline_id for spec in specs}:
        baseline_spec = next((spec for spec in EXPERIMENTS if spec.pipeline_id == args.baseline_pipeline_id), None)
        if baseline_spec is None:
            baseline_spec = next((spec for spec in BASELINE_EXPERIMENTS if spec.pipeline_id == args.baseline_pipeline_id), None)
        if baseline_spec is not None:
            specs = [baseline_spec, *specs]

    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_RUN_ROOT / run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = CandidateBatchRunner(bundle, output_dir=output_dir)
    batch = finalize_decisions(runner.run(specs, baseline_pipeline_id=args.baseline_pipeline_id))
    frame = results_to_frame(batch)
    frame.to_csv(output_dir / "results.csv", index=False)
    (output_dir / "results.json").write_text(json.dumps(frame.to_dict(orient="records"), indent=2, default=str), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": batch.created_at,
                "suite": args.suite,
                "baseline_pipeline_id": args.baseline_pipeline_id,
                "dataset": bundle.manifest,
                "selected_ids": args.candidate_ids,
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        _render_report(bundle, batch, frame, suite=args.suite, selected_ids=args.candidate_ids),
        encoding="utf-8",
    )

    print(f"Dataset version: {bundle.version}")
    print(f"Results written to: {output_dir}")
    print(f"Report written to: {args.report_path}")
    preview = frame.copy()
    preview["test_total_return_pct"] = preview["test_summary.total_return_pct"]
    preview["test_sharpe_ratio"] = preview["test_summary.sharpe_ratio"]
    preview["test_max_drawdown_pct"] = preview["test_summary.max_drawdown_pct"]
    preview = _ranked_summary(preview)
    print(
        preview[
            ["pipeline_id", "decision", "test_total_return_pct", "test_sharpe_ratio", "test_max_drawdown_pct"]
        ]
        .head(10)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
