#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "docs" / "references" / "papers"
PIPELINE_ROOT = ROOT / "docs" / "pipeline_library"
DETAIL_DIR = PIPELINE_ROOT / "candidates"
DETAIL_README = DETAIL_DIR / "README.md"
INDEX_PATH = PIPELINE_ROOT / "07_candidate_dossiers.md"
REPORT_PATH = ROOT / "reports" / "pipeline_candidate_dossiers_2026-04-14.md"


@dataclass(frozen=True)
class PaperRef:
    title: str
    filename: str
    note: str


@dataclass(frozen=True)
class CodeRef:
    path: str
    note: str


@dataclass(frozen=True)
class CodeGroup:
    title: str
    description: str
    refs: tuple[CodeRef, ...]


@dataclass(frozen=True)
class CandidateDoc:
    pipeline_id: str
    title: str
    family: str
    status: str
    fit: str
    priority: str
    summary: str
    paper_refs: tuple[PaperRef, ...]
    code_groups: tuple[str, ...]
    data_bullets: tuple[str, ...]
    execution_bullets: tuple[str, ...]
    risk_bullets: tuple[str, ...]
    validation_bullets: tuple[str, ...]
    archive_bullets: tuple[str, ...]
    next_step: str


def paper(title: str, filename: str, note: str) -> PaperRef:
    path = REFERENCE_ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing reference PDF: {filename}")
    return PaperRef(title=title, filename=filename, note=note)


def code(path: str, note: str) -> CodeRef:
    full = ROOT / path
    if not full.exists():
        raise FileNotFoundError(f"Missing code anchor: {path}")
    return CodeRef(path=path, note=note)


CODE_GROUPS: dict[str, CodeGroup] = {
    "ops": CodeGroup(
        title="Operational Guardrails",
        description="Shared safety and execution rails used by every candidate before a signal can touch capital.",
        refs=(
            code("src/risk/manager.py", "pre-trade risk gate and position haircut"),
            code("src/brokers/paper.py", "paper-safe broker adapter"),
            code("src/monitoring/health.py", "freshness, heartbeat, and kill-switch checks"),
            code("src/backtest/engine.py", "deterministic offline backtest runner"),
            code("src/backtest/matching_engine.py", "queue-aware synthetic fill model"),
            code("scripts/monitor_health.py", "runtime health loop and stale-data monitor"),
        ),
    ),
    "market_data": CodeGroup(
        title="Market Data and Features",
        description="Point-in-time data loading plus the shared feature builder used by most signal stacks.",
        refs=(
            code("src/ingestion/loader.py", "point-in-time market data loading"),
            code("src/features/builder.py", "technical, lagged, and derived feature construction"),
        ),
    ),
    "regime": CodeGroup(
        title="Regime and State",
        description="Regime detection and state overlays used when the signal should shrink or route rather than trade blind.",
        refs=(
            code("src/models/hmm.py", "latent regime detector"),
        ),
    ),
    "ranker": CodeGroup(
        title="Cross-Sectional Ranker",
        description="The executable ranking path that turns scores into a tradable long/flat or top-k book.",
        refs=(
            code("src/models/lgbm.py", "LightGBM scoring model"),
            code("src/features/builder.py", "feature matrix construction for the ranker"),
        ),
    ),
    "fundamental": CodeGroup(
        title="Fundamental Stack",
        description="Point-in-time fundamentals and macro-aware feature joins for slower rebalance cycles.",
        refs=(
            code("src/ingestion/loader.py", "fundamental and macro data loader"),
            code("src/features/builder.py", "feature joins and lag alignment"),
            code("src/models/lgbm.py", "tabular score model for the hybrid stack"),
        ),
    ),
    "news": CodeGroup(
        title="News and Event Stack",
        description="Stream or replay news inputs, event timestamps, and sentiment features for event-aware trading.",
        refs=(
            code("src/ingestion/websocket_client.py", "replay/stream ingestion for event-driven inputs"),
            code("src/ingestion/loader.py", "news or preprocessed corpus loading"),
            code("src/features/builder.py", "event feature construction"),
            code("src/models/lgbm.py", "event classifier or event score model"),
        ),
    ),
    "graph": CodeGroup(
        title="Graph and Correlation Stack",
        description="Correlation graphs and sector relations for diversification-aware ranking.",
        refs=(
            code("src/ingestion/loader.py", "cross-asset data loader"),
            code("src/features/builder.py", "graph and correlation feature construction"),
            code("src/models/lgbm.py", "graph-aware score model"),
        ),
    ),
    "hft": CodeGroup(
        title="HFT and Intraday Stack",
        description="Replay-backed intraday ingestion and synthetic matching for jump filters and short-horizon routing.",
        refs=(
            code("src/ingestion/websocket_client.py", "stream/replay event ingestion"),
            code("src/backtest/matching_engine.py", "intraday queue-aware matching"),
            code("scripts/monitor_health.py", "stream freshness and heartbeat supervision"),
        ),
    ),
}


PAPERS: dict[str, PaperRef] = {
    "BOCD": paper(
        "Bayesian Online Changepoint Detection",
        "0710.3742_bayesian_online_changepoint_detection.pdf",
        "use the changepoint view as a regime prior rather than a full trading engine",
    ),
    "MBSTS": paper(
        "Multivariate Bayesian Structural Time Series Model",
        "1801.03222_multivariate_bsts.pdf",
        "gives the structural time-series framing for trend, seasonality, and latent state",
    ),
    "BL": paper(
        "View fusion vis-à-vis a Bayesian interpretation of Black-Litterman for portfolio allocation",
        "2301.13594_black_litterman_fusion.pdf",
        "provides the posterior view-combination logic for allocation and factor blending",
    ),
    "LIGHTGBM": paper(
        "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
        "lightgbm_neurips_2017.pdf",
        "the tabular baseline and the default comparison target for the whole stack",
    ),
    "MEAN_REV_OPT": paper(
        "On the Profitability of Optimal Mean Reversion Trading Strategies",
        "1602.05858_mean_reversion_optimal.pdf",
        "anchors the mean-reversion side of the SDE and momentum-filter candidates",
    ),
    "MEAN_REV_DEADLINES": paper(
        "Mean Reversion Trading with Sequential Deadlines and Transaction Costs",
        "1707.03498_mean_reversion_deadlines.pdf",
        "useful for delayed entries, deadlines, and transaction-cost-aware exits",
    ),
    "HESTON_PIECEWISE": paper(
        "The Heston stochastic volatility model with piecewise constant parameters - efficient calibration and pricing of window barrier options",
        "1805.04704_heston_piecewise_constant.pdf",
        "the volatility-regime reference for timing and position scaling",
    ),
    "HESTON_JUMPS": paper(
        "Parameter Estimation of the Heston Volatility Model with Jumps in the Asset Prices",
        "2211.14814_heston_jumps.pdf",
        "the jump-risk reference for crash filters and volatility overlays",
    ),
    "FINBERT": paper(
        "FinBERT Financial Sentiment Analysis",
        "5_01_FinBERT_Financial_Sentiment_Analysis.pdf",
        "the sentiment baseline for event and news-driven filters",
    ),
    "FINGPT": paper(
        "FinGPT Open Source Financial LLMs",
        "5_02_FinGPT_Open_Source_Financial_LLMs.pdf",
        "the broader finance-LLM reference for text signal extraction",
    ),
    "LLM_SURVEY": paper(
        "Large Language Models in Finance A Survey",
        "5_11_Large_Language_Models_in_Finance_A_Survey.pdf",
        "keeps the event/news pipelines grounded in a current finance-LLM map",
    ),
    "GRAPH_FORECAST": paper(
        "Graph Neural Networks for Financial Forecasting",
        "5_08_Graph_Neural_Networks_for_Financial_Forecasting.pdf",
        "the graph-forecasting reference for correlation-aware factor construction",
    ),
    "PORT_GRAPH": paper(
        "Portfolio Management with Graph Neural Networks",
        "2_09_Portfolio_Management_with_Graph_Neural_Networks.pdf",
        "useful for diversification-aware allocation and graph-conditioned risk overlay",
    ),
    "MACHINE_RISK": paper(
        "Machine Learning Approach to Portfolio Risk",
        "2_10_Machine_Learning_Approach_to_Portfolio_Risk.pdf",
        "the risk-aware portfolio reference for CVaR and overlay-style control",
    ),
    "OPS_SURVEY": paper(
        "Online Portfolio Selection A Survey",
        "2_12_Online_Portfolio_Selection_A_Survey.pdf",
        "provides the online selection backdrop for residual momentum and rebalancing logic",
    ),
}


def p(key: str) -> PaperRef:
    return PAPERS[key]


CANDIDATES: tuple[CandidateDoc, ...] = (
    CandidateDoc(
        pipeline_id="B001",
        title="Bayesian Linear Return Forecaster",
        family="Bayesian",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "A calibration-first return forecaster that prefers a stable posterior over a complex state machine. "
            "It should produce a confidence-aware alpha score that can be blended with the repo's existing regime and risk gates."
        ),
        paper_refs=(
            PaperRef(p("BOCD").title, p("BOCD").filename, "changepoint prior and shrinkage guard"),
            PaperRef(p("MBSTS").title, p("MBSTS").filename, "structural trend and seasonal baseline"),
        ),
        code_groups=("ops", "market_data", "regime"),
        data_bullets=(
            "Use point-in-time returns, OHLCV, and the technical feature bundle from `src/features/builder.py`.",
            "Feed latent regime state from `src/models/hmm.py` as a shrinkage input rather than as a hard trading rule.",
            "Keep the loader aligned to the same freshness rules enforced by `src/risk/manager.py` and `src/monitoring/health.py`.",
        ),
        execution_bullets=(
            "Estimate a posterior mean and interval for next-period return.",
            "Convert posterior confidence into a long/flat or top-k allocation score.",
            "Send the score through the same backtest and paper-safe gates used by the current executable baseline.",
        ),
        risk_bullets=(
            "Reject stale data and low-liquidity windows before the posterior is used.",
            "Do not trade when posterior variance is wide enough to invert the expected edge.",
            "Keep turnover inside the configured cost budget; otherwise the signal stays in research mode.",
        ),
        validation_bullets=(
            "Compare calibration, hit rate, and risk-adjusted return against the current LightGBM baseline.",
            "Check whether the posterior remains stable under regime shifts and during drawdown clusters.",
            "Re-run the existing smoke and backtest path on the same assets used for F001 so the comparison is apples-to-apples.",
        ),
        archive_bullets=(
            "Archive it if the posterior cannot beat the feature baseline after fees.",
            "Archive it if calibration depends on leaked or late-arriving inputs.",
            "Archive it if the model only works with aggressive parameter tuning rather than a stable prior.",
        ),
        next_step="Use this as a calibration benchmark for B007 and B018 rather than as a standalone production candidate.",
    ),
    CandidateDoc(
        pipeline_id="B003",
        title="Bayesian Factor Regression",
        family="Bayesian",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This is the Bayesian version of a factor model: a posterior over factor exposures rather than a single point estimate. "
            "It is best treated as a disciplined cross-sectional allocator with explicit uncertainty around every factor view."
        ),
        paper_refs=(
            PaperRef(p("BL").title, p("BL").filename, "posterior view fusion for factor blending"),
            PaperRef(p("MACHINE_RISK").title, p("MACHINE_RISK").filename, "risk-aware allocation framing"),
        ),
        code_groups=("ops", "fundamental"),
        data_bullets=(
            "Use point-in-time factor libraries, sector buckets, and any lagged fundamentals exposed through `src/ingestion/loader.py`.",
            "Keep covariance and exposure inputs aligned with the feature matrix in `src/features/builder.py`.",
            "Treat regime state as a shrinkage control rather than a source of alpha on its own.",
        ),
        execution_bullets=(
            "Infer posterior factor exposures and convert them into a ranked score or weight vector.",
            "Blend the posterior with the current allocation engine or with a BL-style prior.",
            "Run the resulting weights through the same risk and paper broker gates used elsewhere in the repo.",
        ),
        risk_bullets=(
            "Do not allow point-in-time leakage from factor updates or delayed fundamentals.",
            "Cap concentration when the posterior overweights a single sector or style bucket.",
            "Reject the model if the uncertainty band is so wide that the allocation becomes arbitrary.",
        ),
        validation_bullets=(
            "Compare factor-neutral performance against F002 and the existing LightGBM baseline.",
            "Measure turnover and stability of the posterior exposures across regime changes.",
            "Check that the factor model improves risk-adjusted return rather than just raw return.",
        ),
        archive_bullets=(
            "Archive if factor timing disappears after costs or lag corrections.",
            "Archive if the factor inputs are not available point-in-time.",
            "Archive if the model is less stable than a simpler BL or HRP overlay.",
        ),
        next_step="Use this to benchmark whether Bayesian uncertainty actually helps over the simpler factor rankers in F002 and F015.",
    ),
    CandidateDoc(
        pipeline_id="B006",
        title="Bayesian State Space Price-Vol Model",
        family="Bayesian",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate models price and volatility as a state-space system so that risk sizing can react before the drawdown arrives. "
            "It is a better fit for overlay logic than for pure alpha generation."
        ),
        paper_refs=(
            PaperRef(p("HESTON_PIECEWISE").title, p("HESTON_PIECEWISE").filename, "volatility-regime calibration reference"),
            PaperRef(p("HESTON_JUMPS").title, p("HESTON_JUMPS").filename, "jump-aware volatility reference"),
        ),
        code_groups=("ops", "market_data", "regime"),
        data_bullets=(
            "Build the state from price returns, realized volatility, and the standard feature matrix in `src/features/builder.py`.",
            "Treat the regime detector in `src/models/hmm.py` as a coarse prior when the volatility state becomes unstable.",
            "Keep the loader on point-in-time inputs so the state does not silently learn from future bars.",
        ),
        execution_bullets=(
            "Forecast volatility one or more steps ahead and scale exposure accordingly.",
            "Use the state posterior as a drawdown haircut rather than as a hard entry signal.",
            "Let the same backtest engine compare the overlay against a rolling-vol baseline.",
        ),
        risk_bullets=(
            "Do not trust the model during jump clusters or after data outages.",
            "Reject stale realized-vol inputs and any state estimate with unstable posterior mass.",
            "Keep the overlay conservative: this is a sizing model, not a reason to lever up.",
        ),
        validation_bullets=(
            "Compare volatility forecast error against a rolling-window baseline.",
            "Check whether the overlay reduces tail losses on the same replay fixtures used by the smoke tests.",
            "Measure whether the model improves risk-adjusted return on top of F001/F023.",
        ),
        archive_bullets=(
            "Archive if the overlay does not reduce drawdown or improve realized-vol accuracy.",
            "Archive if the state estimate is more brittle than the existing HMM overlay.",
            "Archive if the volatility signal does not survive transaction costs.",
        ),
        next_step="Use this as the volatility overlay reference for S003 and F041.",
    ),
    CandidateDoc(
        pipeline_id="B007",
        title="Bayesian State Space Regime Detector",
        family="Bayesian",
        status="supporting components implemented",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This is the most operationally useful Bayesian-style candidate in the set because it routes strategies rather than trying to be the full alpha engine. "
            "The current repo already has the HMM and risk-guard plumbing needed to validate the idea in mock/offline mode."
        ),
        paper_refs=(
            PaperRef(p("BOCD").title, p("BOCD").filename, "change-point framing for regime shifts"),
            PaperRef(p("MBSTS").title, p("MBSTS").filename, "latent-state and trend decomposition reference"),
        ),
        code_groups=("ops", "market_data", "regime"),
        data_bullets=(
            "Use price, return, and realized-volatility features plus any macro inputs that can be aligned point-in-time.",
            "Keep the state aligned to the data freshness checks already handled by the health monitor.",
            "Prefer a small number of latent states so the router stays interpretable.",
        ),
        execution_bullets=(
            "Infer a latent regime and map it to a strategy family or exposure haircut.",
            "Prefer regime persistence over rapid switching; the router should avoid flapping.",
            "Let the paper broker and health monitor validate the routing logic in mock-safe mode before any live consideration.",
        ),
        risk_bullets=(
            "Require a minimum persistence window before a regime change can alter exposure.",
            "Treat stale data or broken heartbeat as a reason to freeze the router.",
            "Do not let regime confidence override the risk manager's kill-switch logic.",
        ),
        validation_bullets=(
            "Check regime stability and transition matrices against the current HMM baseline.",
            "Run the smoke path and ensure the router does not create unnecessary churn.",
            "Stratify the backtest by regime and confirm the strategy mix improves drawdown behavior.",
        ),
        archive_bullets=(
            "Archive if the regime switches are noisier than a simple rolling-vol heuristic.",
            "Archive if the router cannot stay stable under the existing replay fixtures.",
            "Archive if the state machine adds complexity without a measurable risk reduction.",
        ),
        next_step="Use this as the regime gate for B001, F023, and F031 before any deeper Bayesian work.",
    ),
    CandidateDoc(
        pipeline_id="B011",
        title="Bayesian Hierarchical Fundamental Model",
        family="Bayesian",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate pushes slower-moving fundamentals through a hierarchical prior so sector, industry, and asset level signals can coexist. "
            "It is a natural fit for monthly or weekly rebalance schedules rather than for intraday trading."
        ),
        paper_refs=(
            PaperRef(p("BL").title, p("BL").filename, "posterior fusion of views across hierarchy levels"),
            PaperRef(p("MBSTS").title, p("MBSTS").filename, "latent structural trend and macro framing"),
        ),
        code_groups=("ops", "fundamental"),
        data_bullets=(
            "Use point-in-time fundamentals, sector labels, and macro series from the shared loader path.",
            "Lag all reporting variables to their true availability date before they reach the feature builder.",
            "Blend fundamentals with technical and regime features only after lag alignment is locked down.",
        ),
        execution_bullets=(
            "Infer sector and asset-level posterior scores and turn them into a rebalance list.",
            "Use a constrained allocation rule so the hierarchy remains readable and bounded.",
            "Run the rebalance only on the configured schedule; this is not a high-turnover signal.",
        ),
        risk_bullets=(
            "Reject any input that violates point-in-time discipline or report-date lag.",
            "Avoid concentration if a single sector dominates the posterior hierarchy.",
            "Keep the model inactive when the reporting calendar makes the data stale.",
        ),
        validation_bullets=(
            "Compare against the fundamental-heavy hybrids F012 and F025.",
            "Test the rebalance schedule on a monthly and a weekly cadence to ensure stability.",
            "Check that the hierarchy improves risk-adjusted allocation rather than simply overfitting sectors.",
        ),
        archive_bullets=(
            "Archive if point-in-time alignment is too fragile to maintain.",
            "Archive if the hierarchy is less stable than a simpler BL or HRP overlay.",
            "Archive if the gain is not visible after costs and rebalance delay.",
        ),
        next_step="Use it as the slow-moving fundamental benchmark for F012 and F025.",
    ),
    CandidateDoc(
        pipeline_id="B018",
        title="Bayesian Black-Litterman with ML Views",
        family="Bayesian",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This is the most natural place to blend machine-learned views into a portfolio prior without pretending the ML score is certainty. "
            "The Bayesian layer should dominate when the view is weak and relax only when the evidence is stable."
        ),
        paper_refs=(
            PaperRef(p("BL").title, p("BL").filename, "posterior view fusion for the core allocation step"),
            PaperRef(p("LIGHTGBM").title, p("LIGHTGBM").filename, "ML view generator and tabular baseline"),
        ),
        code_groups=("ops", "fundamental", "ranker"),
        data_bullets=(
            "Use cross-sectional factor data, technicals, and any macro or fundamental views that can be aligned point-in-time.",
            "Build the ML view on the same feature matrix used by the existing LightGBM path.",
            "Keep the posterior inputs explicit so the allocation can be audited later.",
        ),
        execution_bullets=(
            "Generate a set of ML views and fuse them with a Bayesian prior into portfolio weights.",
            "Clip overconfident views and cap the influence of highly correlated signals.",
            "Pass the resulting weights through the same risk and execution gates as every other candidate.",
        ),
        risk_bullets=(
            "Reject any ML view that is not backed by point-in-time data.",
            "Limit the influence of correlated views so the posterior does not overfit one factor cluster.",
            "Do not let the allocator become more volatile than the baseline it is supposed to improve.",
        ),
        validation_bullets=(
            "Compare the posterior portfolio to F002 and F012 on both return and concentration metrics.",
            "Track how often the posterior changes weights compared with the ML-only view.",
            "Measure whether the Bayesian fusion improves stability without sacrificing too much upside.",
        ),
        archive_bullets=(
            "Archive if the ML views overpower the prior and turn the model into a glorified ranker.",
            "Archive if turnover or concentration becomes worse than the baseline.",
            "Archive if the posterior cannot be interpreted after a few rebalance cycles.",
        ),
        next_step="Use this as the canonical example of a Bayesian overlay on top of an ML ranker.",
    ),
    CandidateDoc(
        pipeline_id="S003",
        title="Stochastic Volatility Timing Model",
        family="SDE",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate uses a volatility state model as a timing overlay so the strategy can scale down before the pain arrives. "
            "It is most useful when the rest of the stack already has a signal and needs a risk-aware throttle."
        ),
        paper_refs=(
            PaperRef(p("HESTON_PIECEWISE").title, p("HESTON_PIECEWISE").filename, "volatility regime calibration reference"),
            PaperRef(p("HESTON_JUMPS").title, p("HESTON_JUMPS").filename, "jump-aware volatility reference"),
        ),
        code_groups=("ops", "market_data", "regime"),
        data_bullets=(
            "Use returns, realized volatility, and rolling drawdown features from the shared feature builder.",
            "Keep the volatility state aligned with the same freshness rules as the rest of the market data stack.",
            "Let the regime detector shrink exposure whenever the volatility posterior becomes unstable.",
        ),
        execution_bullets=(
            "Forecast the next volatility regime and convert it into a position-size multiplier.",
            "Treat the result as a timing overlay instead of a directional alpha.",
            "Compare the overlay directly against the existing rolling-vol risk control in the backtest engine.",
        ),
        risk_bullets=(
            "Freeze the overlay when jumps or data gaps make the volatility estimate untrustworthy.",
            "Avoid letting a single bad volatility estimate trigger a large exposure jump.",
            "Do not use the model to lever up; it should only justify tighter or looser exposure.",
        ),
        validation_bullets=(
            "Check whether the overlay lowers max drawdown and tail loss on the same replay set used by smoke tests.",
            "Measure volatility forecast error versus a rolling historical baseline.",
            "Confirm the overlay improves risk-adjusted return after transaction costs.",
        ),
        archive_bullets=(
            "Archive if the overlay does not improve drawdown or realized-vol accuracy.",
            "Archive if the state estimate is more brittle than the existing HMM risk gate.",
            "Archive if the signal is only useful in hindsight.",
        ),
        next_step="Use this as the volatility overlay reference for S011 and F041.",
    ),
    CandidateDoc(
        pipeline_id="S004",
        title="Jump Diffusion Crash Filter",
        family="SDE",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate is not trying to predict every move; it is trying to keep the book alive when jumps make the normal assumptions wrong. "
            "Think of it as a crash filter and exposure governor rather than a pure alpha engine."
        ),
        paper_refs=(
            PaperRef(p("HESTON_JUMPS").title, p("HESTON_JUMPS").filename, "jump-risk calibration reference"),
            PaperRef(p("MEAN_REV_DEADLINES").title, p("MEAN_REV_DEADLINES").filename, "deadline and transaction-cost discipline"),
        ),
        code_groups=("ops", "hft", "market_data"),
        data_bullets=(
            "Use gap features, overnight returns, and intraday range statistics from the market-data path.",
            "Replay or stream intraday inputs through `src/ingestion/websocket_client.py` when available.",
            "Keep the filter aligned with the same state and freshness checks enforced by the health monitor.",
        ),
        execution_bullets=(
            "Detect jump probability and immediately haircut exposure when the threshold trips.",
            "Use the crash filter to pause liquidity taking or widen the entry threshold.",
            "Replay the filter with queue-aware synthetic fills so the slippage effect is visible.",
        ),
        risk_bullets=(
            "Prefer false negatives over false positives only when the opportunity cost is explicit; otherwise stay conservative.",
            "Hard-stop trading if the stream or heartbeat breaks during a jump cluster.",
            "Do not let the filter become a de facto signal generator; its job is to defend capital.",
        ),
        validation_bullets=(
            "Measure event-day drawdown, slippage, and false-positive rate on the captured replay fixture.",
            "Check whether the filter cuts risk on crash days without flattening normal periods.",
            "Compare against a simple volatility spike heuristic to avoid unnecessary complexity.",
        ),
        archive_bullets=(
            "Archive if it only helps in hindsight.",
            "Archive if the filter reduces too much alpha in normal markets.",
            "Archive if intraday data quality is not good enough to keep the classifier honest.",
        ),
        next_step="Use this as the crash-risk overlay for S021 and F027.",
    ),
    CandidateDoc(
        pipeline_id="S011",
        title="Mean-Reverting Volatility Allocation Model",
        family="SDE",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate treats volatility itself as a mean-reverting process and allocates accordingly. "
            "It is a sizing tool first and an alpha source second."
        ),
        paper_refs=(
            PaperRef(p("MEAN_REV_OPT").title, p("MEAN_REV_OPT").filename, "mean-reversion process reference"),
            PaperRef(p("MEAN_REV_DEADLINES").title, p("MEAN_REV_DEADLINES").filename, "transaction-cost and deadline discipline"),
        ),
        code_groups=("ops", "market_data", "regime"),
        data_bullets=(
            "Use realized volatility, rolling spread, and price deviation features from the shared market loader.",
            "Keep the state aligned to point-in-time inputs and the same freshness thresholds used by the risk manager.",
            "Treat regime state as a check on how aggressively the mean-reversion overlay can scale exposure.",
        ),
        execution_bullets=(
            "Estimate the current volatility state and scale the allocation toward the historical mean only when the posterior is confident.",
            "Use the output as a volatility-sized allocation overlay rather than as a standalone trade trigger.",
            "Compare the allocation to a simple rolling-vol targeting rule in the backtest engine.",
        ),
        risk_bullets=(
            "Reject the overlay when volatility regimes break or the input stream becomes stale.",
            "Keep the size changes smooth enough that transaction costs do not erase the benefit.",
            "Never let the model increase exposure simply because the historical mean is attractive.",
        ),
        validation_bullets=(
            "Check whether the model improves drawdown and Sharpe relative to a rolling-vol heuristic.",
            "Measure the stability of the mean-reversion signal across calm and stressed periods.",
            "Confirm the overlay still works after cost and slippage assumptions are added.",
        ),
        archive_bullets=(
            "Archive if the mean-reversion effect disappears once costs are included.",
            "Archive if the model is overly sensitive to the calibration window.",
            "Archive if the volatility state is less robust than the existing HMM path.",
        ),
        next_step="Use this as the volatility-sizing benchmark for F041 and the rest of the overlay stack.",
    ),
    CandidateDoc(
        pipeline_id="S021",
        title="Jump Filter + Momentum Engine",
        family="SDE",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate is a momentum engine with a jump filter in front of it. "
            "The point is to stay out of the market when the path assumptions are broken and only let momentum breathe when the jump risk is manageable."
        ),
        paper_refs=(
            PaperRef(p("BOCD").title, p("BOCD").filename, "change-point detector for jump-risk gating"),
            PaperRef(p("HESTON_JUMPS").title, p("HESTON_JUMPS").filename, "jump-risk and volatility reference"),
        ),
        code_groups=("ops", "hft", "market_data"),
        data_bullets=(
            "Use gap features, realized-volatility spikes, and price momentum from the shared feature builder.",
            "Replay intraday or end-of-day bars through the same ingestion path so the filter can be tested in both modes.",
            "Keep the market data clean enough that the jump filter is reacting to jumps, not to stale samples.",
        ),
        execution_bullets=(
            "Allow momentum trades only when jump risk stays below the threshold.",
            "Run the momentum leg through queue-aware synthetic fills when the trade is intraday.",
            "Use the health loop to freeze the engine whenever stream freshness breaks.",
        ),
        risk_bullets=(
            "Be conservative when the market transitions into a jump regime; the engine should prefer flat over forced exposure.",
            "Avoid re-entering too quickly after a jump cluster.",
            "Let the risk manager override the momentum signal whenever drawdown or stale-data checks fail.",
        ),
        validation_bullets=(
            "Measure how much drawdown the jump filter removes on stressed sessions compared with raw momentum.",
            "Check whether the filter degrades normal-period performance less than a naive volatility spike rule.",
            "Run the captured replay fixture and verify the engine does not oscillate around the threshold.",
        ),
        archive_bullets=(
            "Archive if the filter removes more alpha than it saves in risk.",
            "Archive if jump detection is not better than a simple range-break rule.",
            "Archive if the model cannot be replayed deterministically.",
        ),
        next_step="Use it as the crash-aware momentum template for F027 and any future intraday trend stack.",
    ),
    CandidateDoc(
        pipeline_id="F001",
        title="Technical Indicator + LightGBM Ranker",
        family="Financial + DL",
        status="direct code path",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This is the repo's current executable baseline: technical features, a LightGBM scorer, and the existing risk/backtest rails. "
            "It is not glamorous, but it is the highest-confidence path for proving the stack works end-to-end."
        ),
        paper_refs=(
            PaperRef(p("LIGHTGBM").title, p("LIGHTGBM").filename, "core tabular ranker baseline"),
            PaperRef(p("BOCD").title, p("BOCD").filename, "regime prior if the technical score needs a shrinkage overlay"),
        ),
        code_groups=("ops", "market_data", "ranker"),
        data_bullets=(
            "Use OHLCV, returns, and technical indicators from `src/features/builder.py`.",
            "Keep the data pipeline point-in-time and asset-aligned before the ranker sees it.",
            "Use the same risk features the current backtest and smoke paths already validate.",
        ),
        execution_bullets=(
            "Rank assets cross-sectionally and turn the top scores into a long/flat or top-k book.",
            "Let the risk manager haircut exposure whenever volatility or drawdown rules trip.",
            "Replay the same score through offline backtest, synthetic matching, and paper-safe routing.",
        ),
        risk_bullets=(
            "Reject stale bars, duplicated candles, and any feature leak from the future.",
            "Keep the signal within the configured turnover budget.",
            "Stop trading if the health monitor or broker heartbeat fails.",
        ),
        validation_bullets=(
            "Compare directly against the current smoke and backtest paths.",
            "Check that the ranker still works after fees, slippage, and paper-safe gating are applied.",
            "Use the same assets and rebalance cadence when you compare this to F023 and F031.",
        ),
        archive_bullets=(
            "Archive if a simpler technical rule performs better after cost.",
            "Archive if the feature set leaks or becomes hard to maintain.",
            "Archive if the ranker cannot beat the baseline in both return and risk terms.",
        ),
        next_step="Keep this as the benchmark that every more complex candidate must beat.",
    ),
    CandidateDoc(
        pipeline_id="F002",
        title="Factor + LightGBM Cross-Sectional Model",
        family="Financial + DL",
        status="supporting components implemented",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This is the first serious step up from pure technicals: factor exposure, a tabular ranker, and a portfolio-aware allocation rule. "
            "It should be the default comparison point for any factor model in this repo."
        ),
        paper_refs=(
            PaperRef(p("LIGHTGBM").title, p("LIGHTGBM").filename, "the ranker that turns factor rows into scores"),
            PaperRef(p("BL").title, p("BL").filename, "posterior view combination for factor blending"),
        ),
        code_groups=("ops", "fundamental", "ranker"),
        data_bullets=(
            "Use a point-in-time factor library and the shared loader before the score matrix is built.",
            "Keep sector, style, and macro features aligned through `src/features/builder.py`.",
            "Respect the same freshness and slippage assumptions the operational path already enforces.",
        ),
        execution_bullets=(
            "Rank the cross-section with LightGBM and allocate using the existing portfolio/risk rails.",
            "Let the model behave like a disciplined score generator rather than a black-box allocator.",
            "Compare the score to a BL-style posterior when factors overlap heavily.",
        ),
        risk_bullets=(
            "Do not let point-in-time factor availability slip.",
            "Avoid concentration in a single style bucket.",
            "Freeze the model whenever the factor data is stale or the heartbeat is broken.",
        ),
        validation_bullets=(
            "Compare against F001 for raw return and concentration improvement.",
            "Check whether the factor score actually improves cross-sectional dispersion.",
            "Verify the model remains stable under different rebalance cadences.",
        ),
        archive_bullets=(
            "Archive if factor data is too noisy or too delayed to keep in production.",
            "Archive if the ranker never improves over F001 after costs.",
            "Archive if the factor overlay becomes hard to interpret.",
        ),
        next_step="Use this as the factor-aware baseline for F015 and F025.",
    ),
    CandidateDoc(
        pipeline_id="F009",
        title="News Sentiment + LightGBM Event Model",
        family="Financial + DL",
        status="supporting components implemented",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate turns news and sentiment into an event-aware alpha filter rather than assuming text should trade directly. "
            "The safest version uses the text as a gate and lets the LightGBM score decide how much conviction is left."
        ),
        paper_refs=(
            PaperRef(p("FINBERT").title, p("FINBERT").filename, "finance-sentiment baseline"),
            PaperRef(p("FINGPT").title, p("FINGPT").filename, "finance-LLM event extraction"),
            PaperRef(p("LLM_SURVEY").title, p("LLM_SURVEY").filename, "finance-LLM landscape reference"),
        ),
        code_groups=("ops", "news", "ranker"),
        data_bullets=(
            "Use a news corpus or replay stream plus the event timestamps that line up with price data.",
            "Convert article sentiment into a point-in-time feature so late headlines do not leak into the future.",
            "Join the event features with the same technical and risk matrix used by the ranker path.",
        ),
        execution_bullets=(
            "Raise or lower conviction based on the event score instead of entering blindly on the text alone.",
            "Let the LightGBM score absorb the sentiment as one more feature rather than as the entire trade.",
            "Route the event path through the same paper-safe and health-checked execution rail as F001.",
        ),
        risk_bullets=(
            "Deduplicate articles and reject stale news timestamps.",
            "Do not let delayed headlines create hidden lookahead.",
            "Use the health loop to freeze the event engine when the stream freshness or heartbeat degrades.",
        ),
        validation_bullets=(
            "Test on event days versus non-event days and compare against F001.",
            "Measure whether the sentiment feature improves drawdown more than it adds noise.",
            "Check the false-positive rate on neutral or repetitive news cycles.",
        ),
        archive_bullets=(
            "Archive if sentiment only adds lag.",
            "Archive if duplicate news handling becomes too brittle.",
            "Archive if the model is weaker than a plain technical ranker after fees.",
        ),
        next_step="Use this as the event-aware template for F027.",
    ),
    CandidateDoc(
        pipeline_id="F012",
        title="Fundamental + Technical + LightGBM Hybrid",
        family="Financial + DL",
        status="supporting components implemented",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This is the most sensible hybrid for slower capital: fundamentals, technicals, and a tabular ranker. "
            "The key is to keep the fundamentals point-in-time and the technicals strictly lagged."
        ),
        paper_refs=(
            PaperRef(p("LIGHTGBM").title, p("LIGHTGBM").filename, "the hybrid scorer"),
            PaperRef(p("BL").title, p("BL").filename, "view fusion for combining slow and fast signals"),
            PaperRef(p("MBSTS").title, p("MBSTS").filename, "macro and latent-state context for fundamentals"),
        ),
        code_groups=("ops", "fundamental", "ranker"),
        data_bullets=(
            "Join lagged technical features with point-in-time fundamentals and macro inputs.",
            "Keep the feature builder honest about report dates and any delayed availability.",
            "Treat the hybrid as a score generator, not as a source of unchecked conviction.",
        ),
        execution_bullets=(
            "Build a hybrid score and rank the cross-section or ETF universe.",
            "Let risk controls decide how much of the score survives the final allocation.",
            "Test on a monthly or weekly rebalance schedule so the fundamental lag is realistic.",
        ),
        risk_bullets=(
            "Protect against report lag, revision drift, and stale corporate data.",
            "Avoid overreacting to the technical leg when the fundamental inputs are unchanged.",
            "Keep the signal under the same paper/live safety gates as F001.",
        ),
        validation_bullets=(
            "Compare to F001 and F002 on both return and concentration.",
            "Stratify the result by rebalance frequency to verify the lag assumptions.",
            "Check whether the hybrid improves stability without becoming too slow.",
        ),
        archive_bullets=(
            "Archive if the fundamental lag is too costly to manage.",
            "Archive if the hybrid is no better than the simpler rankers after costs.",
            "Archive if the model becomes sensitive to reporting windows.",
        ),
        next_step="Use this as the canonical hybrid baseline for F025 and F039.",
    ),
    CandidateDoc(
        pipeline_id="F015",
        title="Residual Momentum + LightGBM",
        family="Financial + DL",
        status="supporting components implemented",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate tries to remove the obvious factor effects first and then rank the residual momentum. "
            "That makes it a better test of whether there is any genuine idiosyncratic trend left after the easy stuff is stripped away."
        ),
        paper_refs=(
            PaperRef(p("LIGHTGBM").title, p("LIGHTGBM").filename, "the residual ranker"),
            PaperRef(p("OPS_SURVEY").title, p("OPS_SURVEY").filename, "online selection and rebalance framing"),
        ),
        code_groups=("ops", "market_data", "ranker"),
        data_bullets=(
            "Use the factor library to residualize raw returns before they reach the ranker.",
            "Keep the technical and lagged feature set aligned with the same point-in-time source.",
            "Make the residual step explicit so leakage cannot hide inside the score matrix.",
        ),
        execution_bullets=(
            "Rank the residual momentum score and allocate only to the strongest names.",
            "Use the same risk manager and backtest engine to compare against raw momentum.",
            "If the residual leg is weak, let the system fall back to the simpler ranker instead of forcing exposure.",
        ),
        risk_bullets=(
            "Do not let factor residualization leak future information.",
            "Be explicit about the factor universe; if it is not maintained, the residual score is not valid.",
            "Drop the signal if costs eat the residual edge.",
        ),
        validation_bullets=(
            "Compare factor-neutral return and drawdown against F001 and F002.",
            "Check whether the residual score remains stable across different universes.",
            "Verify that the residual step improves signal quality rather than merely changing turnover.",
        ),
        archive_bullets=(
            "Archive if factor-neutralization removes the entire edge.",
            "Archive if the score is too unstable to maintain.",
            "Archive if the residual workflow becomes a maintenance burden.",
        ),
        next_step="Use this to test whether there is any true residual alpha beyond the factor baseline.",
    ),
    CandidateDoc(
        pipeline_id="F019",
        title="Correlation Graph + LightGBM Overlay",
        family="Financial + DL",
        status="supporting components implemented",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate uses correlation structure as an extra context layer on top of the tabular score model. "
            "The graph itself is a research extension, but the operational path remains the same ranker-plus-risk stack."
        ),
        paper_refs=(
            PaperRef(p("GRAPH_FORECAST").title, p("GRAPH_FORECAST").filename, "graph-based financial forecasting reference"),
            PaperRef(p("PORT_GRAPH").title, p("PORT_GRAPH").filename, "graph-conditioned allocation and diversification"),
            PaperRef(p("LIGHTGBM").title, p("LIGHTGBM").filename, "score model sitting on top of the graph features"),
        ),
        code_groups=("ops", "graph", "ranker"),
        data_bullets=(
            "Build correlation or sector graphs from point-in-time cross-asset returns.",
            "Keep graph edges stable enough that the score model is not learning random window noise.",
            "Join graph context with the standard technical and lagged feature set in the builder.",
        ),
        execution_bullets=(
            "Use graph-aware features to adjust the ranker score or the portfolio concentration limit.",
            "Let the graph overlay act as a diversification haircut rather than as a direct trade trigger.",
            "Run the resulting signal through the normal backtest and paper-safe rails.",
        ),
        risk_bullets=(
            "Keep the graph window stable and point-in-time; unstable edges are worse than no graph at all.",
            "Do not let graph features silently overfit one short window.",
            "Respect the same cost and exposure rules that govern F001.",
        ),
        validation_bullets=(
            "Compare concentration, turnover, and drawdown against F001 and F002.",
            "Check whether the graph overlay improves diversification more than it adds instability.",
            "Verify that the graph features survive a holdout regime rather than only the training window.",
        ),
        archive_bullets=(
            "Archive if the graph windows are too noisy to maintain.",
            "Archive if diversification improves only on paper and not after costs.",
            "Archive if the overlay cannot be kept point-in-time.",
        ),
        next_step="Use this as the graph-aware diversification test for future factor work.",
    ),
    CandidateDoc(
        pipeline_id="F023",
        title="Regime Detector + LightGBM Stack",
        family="Financial + DL",
        status="direct code path",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This is the current regime-aware alpha baseline: HMM-style state detection plus the LightGBM ranker. "
            "It is one of the cleanest examples of an executable hybrid in the repo."
        ),
        paper_refs=(
            PaperRef(p("BOCD").title, p("BOCD").filename, "change-point framing for the regime detector"),
            PaperRef(p("LIGHTGBM").title, p("LIGHTGBM").filename, "the alpha model sitting behind the regime gate"),
        ),
        code_groups=("ops", "market_data", "regime", "ranker"),
        data_bullets=(
            "Use the shared market loader and feature builder as the alpha input layer.",
            "Feed the regime detector with price, vol, and any macro features that can be aligned point-in-time.",
            "Keep the ranker input identical to F001 so the regime effect can be isolated cleanly.",
        ),
        execution_bullets=(
            "Detect the regime first and then allow the ranker to trade only in the permitted states.",
            "Route the score through the same risk haircut and backtest engine as the other baselines.",
            "Use the paper broker and health monitor to prove the gate behaves in mock-safe mode.",
        ),
        risk_bullets=(
            "Do not let the gate flip too often; regime noise is worse than a missing regime.",
            "Freeze the model if data freshness or heartbeat checks fail.",
            "Keep the ranker and the regime state on the same clock to avoid hidden leakage.",
        ),
        validation_bullets=(
            "Check regime-stratified return and drawdown against F001.",
            "Measure whether the gate reduces losses during stressed periods without killing normal performance.",
            "Re-run the smoke/backtest path to confirm the gating logic is deterministic.",
        ),
        archive_bullets=(
            "Archive if the gate does not improve drawdown or stability.",
            "Archive if the regime detector becomes noisier than a rolling-vol heuristic.",
            "Archive if the gating logic makes the score harder to maintain than it is worth.",
        ),
        next_step="Use this as the direct executable baseline for regime-aware scoring.",
    ),
    CandidateDoc(
        pipeline_id="F025",
        title="Fundamental Delay-Aware LightGBM Model",
        family="Financial + DL",
        status="supporting components implemented",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate is about enforcing the annoying but necessary truth that fundamentals arrive late and sometimes revise. "
            "The model is only useful if the lag is explicit and the score stays honest about what was actually known at the time."
        ),
        paper_refs=(
            PaperRef(p("LIGHTGBM").title, p("LIGHTGBM").filename, "the tabular score engine"),
            PaperRef(p("MBSTS").title, p("MBSTS").filename, "latent trend and lag-aware state framing"),
            PaperRef(p("BL").title, p("BL").filename, "posterior fusion when fundamentals and technicals overlap"),
        ),
        code_groups=("ops", "fundamental", "ranker"),
        data_bullets=(
            "Keep the fundamental feed point-in-time and apply the actual publication lag before any join.",
            "Blend technicals and fundamentals only after the lagged data is aligned in `src/features/builder.py`.",
            "Use the same cross-sectional universe as F012 so the lag effect can be isolated.",
        ),
        execution_bullets=(
            "Rebalance only when the fundamentals are legitimately available.",
            "Let the LightGBM model score the delayed fundamental snapshot rather than the raw reporting date.",
            "Use the risk manager to ensure the slower cadence does not accidentally expand turnover.",
        ),
        risk_bullets=(
            "Treat any report-date mismatch as a hard failure.",
            "Reject the model if the lag logic becomes too complex to audit.",
            "Do not allow late revisions to leak into historical training labels.",
        ),
        validation_bullets=(
            "Check explicitly for lookahead around announcements and filing dates.",
            "Compare the delayed model to F012 on the same assets and same rebalance cadence.",
            "Measure whether the lagged fundamentals still add value after costs.",
        ),
        archive_bullets=(
            "Archive if the delay logic is too brittle for production use.",
            "Archive if the lag erases the edge after transaction costs.",
            "Archive if the report-date handling is not maintainable.",
        ),
        next_step="Use this as the canonical test of whether fundamental delay handling is robust enough for live use.",
    ),
    CandidateDoc(
        pipeline_id="F026",
        title="Technical + Factor + XGBoost with CVaR Overlay",
        family="Financial + DL",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate is the more aggressive hybrid: technicals, factors, a boosted tree stack, and a risk overlay that explicitly cares about the left tail. "
            "It should only survive if the risk layer can justify the extra complexity."
        ),
        paper_refs=(
            PaperRef(p("LIGHTGBM").title, p("LIGHTGBM").filename, "tabular baseline for the boosted-tree family"),
            PaperRef(p("MACHINE_RISK").title, p("MACHINE_RISK").filename, "CVaR and risk-aware allocation framing"),
            PaperRef(p("BL").title, p("BL").filename, "posterior fusion for blending technical and factor views"),
        ),
        code_groups=("ops", "fundamental", "ranker"),
        data_bullets=(
            "Use technicals, factors, and any lagged macro features that can be joined point-in-time.",
            "Keep the factor library and the feature builder aligned so the boosted tree does not inherit leakage.",
            "Treat the CVaR overlay as a risk gate, not as a reason to grow the signal.",
        ),
        execution_bullets=(
            "Generate a boosted-tree score and pass it through a CVaR-aware allocation rule.",
            "Keep the overlay conservative when the left tail widens.",
            "Use the same backtest engine and paper-safe path to compare against F001 and F012.",
        ),
        risk_bullets=(
            "Do not let optimization complexity overwhelm the signal.",
            "Reject any CVaR solution that is numerically unstable or too slow for the rebalance schedule.",
            "Keep the risk overlay explicit and auditable; this is not a black-box hedge fund trick.",
        ),
        validation_bullets=(
            "Compare tail loss, turnover, and solve time against the simpler rankers.",
            "Check whether the CVaR layer actually improves worst-case periods.",
            "Measure whether the boosted tree contributes anything after the risk overlay is applied.",
        ),
        archive_bullets=(
            "Archive if the risk overlay adds more instability than protection.",
            "Archive if the boosted tree does not improve over F001/F012 after costs.",
            "Archive if the optimization step becomes a maintenance burden.",
        ),
        next_step="Use this as the stress-test version of the hybrid stack and only promote it if the tail improves.",
    ),
    CandidateDoc(
        pipeline_id="F027",
        title="News Shock Filter + Momentum Engine",
        family="Financial + DL",
        status="research candidate",
        fit="H",
        priority="1차 핵심",
        summary=(
            "This candidate is the event-aware complement to F009: instead of just scoring sentiment, it blocks momentum when the news flow says the market is being repriced. "
            "It is a guardrail-first design for noisy headlines and shock-heavy sessions."
        ),
        paper_refs=(
            PaperRef(p("FINBERT").title, p("FINBERT").filename, "sentiment baseline for event classification"),
            PaperRef(p("FINGPT").title, p("FINGPT").filename, "finance-LLM event extraction"),
            PaperRef(p("LLM_SURVEY").title, p("LLM_SURVEY").filename, "current finance-LLM context"),
            PaperRef(p("BOCD").title, p("BOCD").filename, "change-point framing for shock detection"),
        ),
        code_groups=("ops", "news", "hft"),
        data_bullets=(
            "Use news, sentiment, and price context on the same event clock.",
            "Join the shock features with the intraday or end-of-day market data through the shared feature builder.",
            "Keep the replay stream available so the filter can be exercised without a live broker.",
        ),
        execution_bullets=(
            "Allow momentum trades only when the shock filter stays calm.",
            "Use the event score to cut exposure before the momentum leg fires.",
            "Route the decision through the same paper-safe and health-checked path used by the rest of the repo.",
        ),
        risk_bullets=(
            "Be conservative about duplicate articles and delayed headlines.",
            "Do not let macro headlines masquerade as stock-specific shocks.",
            "Freeze the engine when stream freshness or broker heartbeat is broken.",
        ),
        validation_bullets=(
            "Compare event-day drawdown and turnover against F009 and F001.",
            "Check whether the filter reduces crash exposure without destroying normal momentum performance.",
            "Replay the captured fixture and confirm the filter stays deterministic across replays.",
        ),
        archive_bullets=(
            "Archive if the shock filter is noisier than the sentiment score it was supposed to improve.",
            "Archive if duplicate news handling or stale headlines break the logic.",
            "Archive if the filter becomes too conservative to trade.",
        ),
        next_step="Use this as the news-shock template for a future intraday event router.",
    ),
)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "candidate"


def candidate_filename(candidate: CandidateDoc) -> str:
    return f"{candidate.pipeline_id}_{slugify(candidate.title)}.md"


def link_paper(filename: str, title: str) -> str:
    return f"[{title}](../../references/papers/{filename})"


def link_code(path: str) -> str:
    return f"[{path}](../../../{path})"


def render_bullets(items: Iterable[str]) -> list[str]:
    return [f"- {item}" for item in items]


def render_code_groups(group_keys: Iterable[str]) -> list[str]:
    lines: list[str] = []
    for key in group_keys:
        group = CODE_GROUPS[key]
        lines.extend([f"### {group.title}", "", group.description])
        for ref in group.refs:
            lines.append(f"- {link_code(ref.path)} - {ref.note}")
        lines.append("")
    return lines


def render_papers(papers: Iterable[PaperRef]) -> list[str]:
    return [f"- {link_paper(ref.filename, ref.title)} - {ref.note}" for ref in papers]


def render_candidate(candidate: CandidateDoc) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        f"# {candidate.pipeline_id}. {candidate.title}",
        "",
        "<!-- generated by scripts/generate_candidate_pipeline_docs.py -->",
        "",
        f"- source: `05_priority_shortlist.md`",
        f"- generated_at: `{generated_at}`",
        f"- parent_index: [Candidate Dossier Index](../07_candidate_dossiers.md)",
        f"- scope: `first 20 primary shortlist candidates`",
        "",
        "## Spec Card",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Family | {candidate.family} |",
        f"| Status | {candidate.status} |",
        f"| Fit | {candidate.fit} |",
        f"| Priority | {candidate.priority} |",
        f"| Paper Count | {len(candidate.paper_refs)} |",
        f"| Code Groups | {', '.join(candidate.code_groups)} |",
        "",
        "## Positioning",
        "",
        candidate.summary,
        "",
        "## Reference Basis",
        "",
        *render_papers(candidate.paper_refs),
        "",
        "## Repo Anchors",
        "",
        *render_code_groups(candidate.code_groups),
        "## Data / Feature Bundle",
        "",
        *render_bullets(candidate.data_bullets),
        "",
        "## Execution Path",
        "",
        *render_bullets(candidate.execution_bullets),
        "",
        "## Risk Controls",
        "",
        *render_bullets(candidate.risk_bullets),
        "",
        "## Validation Gates",
        "",
        *render_bullets(candidate.validation_bullets),
        "",
        "## Archive Triggers",
        "",
        *render_bullets(candidate.archive_bullets),
        "",
        "## Next Step",
        "",
        candidate.next_step,
        "",
    ]
    return "\n".join(lines)


def render_index(candidates: Iterable[CandidateDoc]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    by_family: dict[str, list[CandidateDoc]] = {}
    for candidate in candidates:
        by_family.setdefault(candidate.family, []).append(candidate)

    lines = [
        "# Candidate Dossier Index",
        "",
        "<!-- generated by scripts/generate_candidate_pipeline_docs.py -->",
        "",
        f"- generated_at: `{generated_at}`",
        "- scope: first 20 primary shortlist candidates from `05_priority_shortlist.md`",
        "- detail_dir: `docs/pipeline_library/candidates/`",
        "",
        "## How to Read",
        "",
        "- Start with `F001`, `F023`, and `F031` if you want the executable baseline first.",
        "- Then read the Bayesian and SDE overlays that improve calibration, risk, and regime handling.",
        "- The docs are intentionally practical: each one includes data, execution, risk, validation, and archive criteria.",
        "- For folder navigation, open [`Candidate Folder README`](./README.md).",
        "",
    ]

    family_order = ["Financial + DL", "Bayesian", "SDE"]
    for family in family_order:
        if family not in by_family:
            continue
        lines.extend([f"## {family}", "", "| ID | Name | Status | Fit | Doc | Reference Basis |", "| --- | --- | --- | --- | --- | --- |"])
        for candidate in by_family[family]:
            docs_rel = f"./candidates/{candidate_filename(candidate)}"
            paper_names = ", ".join(ref.title for ref in candidate.paper_refs[:2])
            if len(candidate.paper_refs) > 2:
                paper_names += ", ..."
            lines.append(
                f"| {candidate.pipeline_id} | {candidate.title} | {candidate.status} | {candidate.fit} | "
                f"[doc]({docs_rel}) | {paper_names} |"
            )
        lines.append("")

    return "\n".join(lines)


def render_detail_readme(candidates: Iterable[CandidateDoc]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    candidates = list(candidates)
    family_counts = {
        family: sum(1 for candidate in candidates if candidate.family == family)
        for family in ("Bayesian", "SDE", "Financial + DL")
    }
    status_counts = {
        status: sum(1 for candidate in candidates if candidate.status == status)
        for status in (
            "direct code path",
            "supporting components implemented",
            "research candidate",
        )
        if any(candidate.status == status for candidate in candidates)
    }

    lines = [
        "# Candidate Dossiers",
        "",
        "<!-- generated by scripts/generate_candidate_pipeline_docs.py -->",
        "",
        f"- generated_at: `{generated_at}`",
        "- manifest: [`manifest.json`](./manifest.json)",
        "- index: [`Candidate Dossier Index`](../07_candidate_dossiers.md)",
        f"- scope: `{len(candidates)}` detailed primary shortlist dossiers",
        "",
        "## Purpose",
        "",
        "- This folder contains the long-form dossiers referenced by the pipeline library.",
        "- Use this README for folder browsing and the index for curated reading order.",
        "- Each dossier is grounded in the downloaded reference papers under `docs/references/papers/`.",
        "",
        "## Snapshot",
        "",
        f"- family_counts: Bayesian={family_counts['Bayesian']}, SDE={family_counts['SDE']}, Financial + DL={family_counts['Financial + DL']}",
        "- status_counts: " + ", ".join(f"{status}={count}" for status, count in status_counts.items()),
        "",
        "## Folder Map",
        "",
        "| ID | Name | Family | Status | Doc | Key References |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for candidate in candidates:
        ref_titles = ", ".join(ref.title for ref in candidate.paper_refs[:2])
        if len(candidate.paper_refs) > 2:
            ref_titles = f"{ref_titles}, ..."
        lines.append(
            "| {id} | {name} | {family} | {status} | [doc](./{doc}) | {refs} |".format(
                id=candidate.pipeline_id,
                name=candidate.title,
                family=candidate.family,
                status=candidate.status,
                doc=candidate_filename(candidate),
                refs=ref_titles,
            )
        )

    lines.extend(
        [
            "",
            "## How To Read",
            "",
            "- Start with the executable baseline dossiers: `F001`, `F023`, and `F031`.",
            "- Then move to the Bayesian and SDE overlays that tighten calibration and risk control.",
            "- Use `manifest.json` if you need machine-readable metadata for automation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        filename = candidate_filename(candidate)
        doc_path = DETAIL_DIR / filename
        write_file(doc_path, render_candidate(candidate))
        manifest_rows.append(
            {
                "pipeline_id": candidate.pipeline_id,
                "title": candidate.title,
                "family": candidate.family,
                "status": candidate.status,
                "fit": candidate.fit,
                "priority": candidate.priority,
                "path": str(doc_path.relative_to(ROOT)),
                "papers": [ref.filename for ref in candidate.paper_refs],
                "code_groups": list(candidate.code_groups),
            }
        )

    write_file(DETAIL_README, render_detail_readme(CANDIDATES))
    write_file(INDEX_PATH, render_index(CANDIDATES))

    report_lines = [
        "# Candidate Pipeline Dossiers Report",
        "",
        "Date: 2026-04-14",
        "",
        "## What Was Generated",
        "",
        f"- Detailed candidate docs: `{len(CANDIDATES)}`",
        f"- Index file: `{INDEX_PATH.relative_to(ROOT)}`",
        f"- Detail directory: `{DETAIL_DIR.relative_to(ROOT)}`",
        f"- Folder README: `{DETAIL_README.relative_to(ROOT)}`",
        "",
        "## Coverage",
        "",
        "- These dossiers cover the first 20 primary candidates from `05_priority_shortlist.md`.",
        "- The set is intentionally focused on the highest-confidence Bayesian, SDE, and Financial + DL paths.",
        "- Every doc includes reference papers, repo anchors, execution path, risk controls, validation gates, and archive triggers.",
        "",
        "## Next Step",
        "",
        "- Re-run `python scripts/generate_pipeline_docs.py` so the shortlist and catalog link back to this index.",
        "- Add more dossiers later if you want the 2차 RL candidate set or the remaining research-only items.",
        "",
    ]
    write_file(REPORT_PATH, "\n".join(report_lines))

    manifest_path = DETAIL_DIR / "manifest.json"
    write_file(manifest_path, json.dumps(manifest_rows, ensure_ascii=False, indent=2))
    print(f"Generated {len(CANDIDATES)} candidate docs, index, report, and manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
