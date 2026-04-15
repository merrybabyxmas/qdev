from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REFERENCE_SPECS = [
    {
        "id": "bochd",
        "title": "Bayesian Online Changepoint Detection",
        "year": 2007,
        "tags": ["bayesian", "regime", "monitoring"],
        "source_url": "https://arxiv.org/pdf/0710.3742.pdf",
        "filename": "0710.3742_bayesian_online_changepoint_detection.pdf",
    },
    {
        "id": "mbsts",
        "title": "Multivariate Bayesian Structural Time Series Model",
        "year": 2018,
        "tags": ["bayesian", "macro", "regime"],
        "source_url": "https://arxiv.org/pdf/1801.03222.pdf",
        "filename": "1801.03222_multivariate_bsts.pdf",
    },
    {
        "id": "black_litterman_fusion",
        "title": "View fusion vis-à-vis a Bayesian interpretation of Black-Litterman for portfolio allocation",
        "year": 2023,
        "tags": ["bayesian", "portfolio"],
        "source_url": "https://arxiv.org/pdf/2301.13594.pdf",
        "filename": "2301.13594_black_litterman_fusion.pdf",
    },
    {
        "id": "mean_reversion_optimal",
        "title": "On the Profitability of Optimal Mean Reversion Trading Strategies",
        "year": 2016,
        "tags": ["sde", "mean_reversion", "pairs"],
        "source_url": "https://arxiv.org/pdf/1602.05858.pdf",
        "filename": "1602.05858_mean_reversion_optimal.pdf",
    },
    {
        "id": "mean_reversion_deadlines",
        "title": "Mean Reversion Trading with Sequential Deadlines and Transaction Costs",
        "year": 2017,
        "tags": ["sde", "mean_reversion", "execution"],
        "source_url": "https://arxiv.org/pdf/1707.03498.pdf",
        "filename": "1707.03498_mean_reversion_deadlines.pdf",
    },
    {
        "id": "heston_piecewise",
        "title": "The Heston stochastic volatility model with piecewise constant parameters - efficient calibration and pricing of window barrier options",
        "year": 2018,
        "tags": ["sde", "volatility"],
        "source_url": "https://arxiv.org/pdf/1805.04704.pdf",
        "filename": "1805.04704_heston_piecewise_constant.pdf",
    },
    {
        "id": "heston_jumps",
        "title": "Parameter Estimation of the Heston Volatility Model with Jumps in the Asset Prices",
        "year": 2022,
        "tags": ["sde", "volatility", "jump"],
        "source_url": "https://arxiv.org/pdf/2211.14814.pdf",
        "filename": "2211.14814_heston_jumps.pdf",
    },
    {
        "id": "dqn",
        "title": "Playing Atari with Deep Reinforcement Learning",
        "year": 2013,
        "tags": ["rl"],
        "source_url": "https://arxiv.org/pdf/1312.5602.pdf",
        "filename": "1312.5602_dqn.pdf",
    },
    {
        "id": "ppo",
        "title": "Proximal Policy Optimization Algorithms",
        "year": 2017,
        "tags": ["rl"],
        "source_url": "https://arxiv.org/pdf/1707.06347.pdf",
        "filename": "1707.06347_ppo.pdf",
    },
    {
        "id": "sac",
        "title": "Soft Actor-Critic Algorithms and Applications",
        "year": 2018,
        "tags": ["rl"],
        "source_url": "https://arxiv.org/pdf/1812.05905.pdf",
        "filename": "1812.05905_sac.pdf",
    },
    {
        "id": "portfolio_drl",
        "title": "Adversarial Deep Reinforcement Learning in Portfolio Management",
        "year": 2018,
        "tags": ["rl", "portfolio"],
        "source_url": "https://arxiv.org/pdf/1808.09940.pdf",
        "filename": "1808.09940_adversarial_drl_portfolio.pdf",
    },
    {
        "id": "model_based_drl",
        "title": "Model-based Deep Reinforcement Learning for Dynamic Portfolio Optimization",
        "year": 2019,
        "tags": ["rl", "portfolio"],
        "source_url": "https://arxiv.org/pdf/1901.08740.pdf",
        "filename": "1901.08740_model_based_drl_portfolio.pdf",
    },
    {
        "id": "lightgbm",
        "title": "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
        "year": 2017,
        "tags": ["financial_dl", "tabular", "baseline"],
        "source_url": "https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf",
        "filename": "lightgbm_neurips_2017.pdf",
    },
    {
        "id": "deeplob",
        "title": "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books",
        "year": 2018,
        "tags": ["financial_dl", "hft"],
        "source_url": "https://arxiv.org/pdf/1808.03668.pdf",
        "filename": "1808.03668_deeplob.pdf",
    },
    {
        "id": "lob_multi_horizon",
        "title": "Multi-Horizon Forecasting for Limit Order Books: Novel Deep Learning Approaches and Hardware Acceleration using Intelligent Processing Units",
        "year": 2021,
        "tags": ["financial_dl", "hft"],
        "source_url": "https://arxiv.org/pdf/2105.10430.pdf",
        "filename": "2105.10430_lob_multi_horizon.pdf",
    },
    {
        "id": "tft",
        "title": "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting",
        "year": 2019,
        "tags": ["financial_dl", "time_series"],
        "source_url": "https://arxiv.org/pdf/1912.09363.pdf",
        "filename": "1912.09363_temporal_fusion_transformer.pdf",
    },
    {
        "id": "lob_market_making_rl",
        "title": "Market Making with Deep Reinforcement Learning from Limit Order Books",
        "year": 2023,
        "tags": ["hft", "rl"],
        "source_url": "https://arxiv.org/pdf/2305.15821.pdf",
        "filename": "2305.15821_market_making_rl_lob.pdf",
    },
    {
        "id": "lob_adaptive_traders",
        "title": "Deep Learning can Replicate Adaptive Traders in a Limit-Order-Book Financial Market",
        "year": 2018,
        "tags": ["hft", "financial_dl"],
        "source_url": "https://arxiv.org/pdf/1811.02880.pdf",
        "filename": "1811.02880_adaptive_traders_lob.pdf",
    },
]


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "docs" / "references"
PAPER_DIR = REFERENCE_DIR / "papers"
MANIFEST_PATH = REFERENCE_DIR / "reference_manifest.json"
INDEX_PATH = REFERENCE_DIR / "index.md"


def _download(url: str, destination: Path, force: bool = False) -> dict[str, object]:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return {
            "status": "exists",
            "bytes": destination.stat().st_size,
            "path": str(destination.relative_to(ROOT)),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    return {
        "status": "downloaded",
        "bytes": destination.stat().st_size,
        "path": str(destination.relative_to(ROOT)),
    }


def _render_index(records: list[dict[str, object]]) -> str:
    lines = [
        "# Reference Index",
        "",
        "<!-- generated by scripts/download_pipeline_references.py -->",
        "",
        "Downloaded paper and archive references used to ground the pipeline docs.",
        "",
        "| ID | Title | Year | Tags | Local PDF | Source | Status |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for record in records:
        pdf_path = record.get("paper_relpath") or f"papers/{record.get('filename', '')}"
        pdf_link = f"[pdf]({pdf_path})" if pdf_path else ""
        tags = ", ".join(record.get("tags", []))
        source = record.get("source_url", "")
        status = record.get("download_status", "pending")
        lines.append(
            f"| {record['id']} | {record['title']} | {record['year']} | {tags} | {pdf_link} | "
            f"[source]({source}) | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download paper and archive references for the pipeline docs.")
    parser.add_argument("--force", action="store_true", help="Re-download PDFs even if they already exist.")
    args = parser.parse_args()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    for spec in REFERENCE_SPECS:
        destination = PAPER_DIR / spec["filename"]
        try:
            result = _download(spec["source_url"], destination, force=args.force)
            record = {
                **spec,
                "local_path": str(destination.relative_to(ROOT)),
                "paper_relpath": f"papers/{spec['filename']}",
                "download_status": result["status"],
                "download_bytes": result["bytes"],
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            }
            records.append(record)
            print(f"{spec['id']}: {result['status']} -> {destination}")
        except Exception as exc:  # pragma: no cover - network dependent
            failures.append((spec["id"], str(exc)))
            records.append(
                {
                    **spec,
                    "local_path": str(destination.relative_to(ROOT)),
                    "download_status": "failed",
                    "download_error": str(exc),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"{spec['id']}: failed ({exc})")

    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    INDEX_PATH.write_text(_render_index(records), encoding="utf-8")

    print(f"Wrote manifest: {MANIFEST_PATH}")
    print(f"Wrote index: {INDEX_PATH}")
    if failures:
        print("Some references failed to download:")
        for ref_id, error in failures:
            print(f"- {ref_id}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
