from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SNAPSHOT_PATH = ROOT / "artifacts" / "control_plane" / "dashboard_snapshot.json"
LEADERBOARD_PATH = ROOT / "artifacts" / "control_plane" / "leaderboard.csv"
LOG_ROOT = ROOT / "artifacts" / "control_plane" / "logs"
SOAK_RECORDS_PATH = ROOT / "artifacts" / "paper_soak" / "soak_records.jsonl"
LEADERBOARD_HISTORY_PATH = ROOT / "artifacts" / "control_plane" / "leaderboard_history.jsonl"


def _python_executable() -> str:
    return str(PYTHON if PYTHON.exists() else Path(sys.executable))


def _run_command(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join((completed.stdout.splitlines() + completed.stderr.splitlines())[-40:])
    return completed.returncode, tail


def _refresh_snapshot() -> tuple[int, str]:
    return _run_command([_python_executable(), str(ROOT / "scripts" / "refresh_control_plane.py")])


def _load_snapshot() -> dict[str, object]:
    if not SNAPSHOT_PATH.exists():
        _refresh_snapshot()
    if not SNAPSHOT_PATH.exists():
        return {}
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _load_leaderboard() -> pd.DataFrame:
    if not LEADERBOARD_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LEADERBOARD_PATH)


def _load_text_tail(path: Path, limit: int = 30) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def _load_jsonl_tail(path: Path, limit: int = 20) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return pd.DataFrame(rows)


def _snapshot_leaderboard_rankings(df: pd.DataFrame) -> None:
    if df.empty:
        return
    needed = {"pipeline_id", "test_summary.total_return_pct", "test_summary.sharpe_ratio", "final_score"}
    if not needed.issubset(df.columns):
        return
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    ranked = df[list(needed)].copy()
    ranked["return_rank"] = ranked["test_summary.total_return_pct"].rank(ascending=False, method="min").astype(int)
    ranked["sharpe_rank"] = ranked["test_summary.sharpe_ratio"].rank(ascending=False, method="min").astype(int)
    ranked["score_rank"] = ranked["final_score"].rank(ascending=False, method="min").astype(int)
    with open(LEADERBOARD_HISTORY_PATH, "a", encoding="utf-8") as f:
        for _, row in ranked.iterrows():
            f.write(json.dumps({
                "timestamp": now,
                "pipeline_id": row["pipeline_id"],
                "return_rank": int(row["return_rank"]),
                "sharpe_rank": int(row["sharpe_rank"]),
                "score_rank": int(row["score_rank"]),
            }) + "\n")


def _load_leaderboard_history() -> pd.DataFrame:
    if not LEADERBOARD_HISTORY_PATH.exists():
        return pd.DataFrame()
    rows = []
    for line in LEADERBOARD_HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _status_badge(value: bool) -> str:
    return "Healthy" if value else "Attention"


HFT_STATUS_PATH = ROOT / "artifacts" / "control_plane" / "hft_status.json"
HFT_TICKS_PATH = ROOT / "artifacts" / "control_plane" / "logs" / "hft_ticks.jsonl"


def _load_hft_status() -> dict | None:
    if not HFT_STATUS_PATH.exists():
        return None
    try:
        return json.loads(HFT_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_hft_ticks(limit: int = 200) -> pd.DataFrame:
    if not HFT_TICKS_PATH.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for line in HFT_TICKS_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


st.set_page_config(page_title="QDev Control Plane", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
      background: radial-gradient(circle at top left, #f6efe4 0%, #f5f2ea 45%, #f2f6f7 100%);
      color: #10222b;
    }
    .hero {
      padding: 1rem 1.25rem;
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(201,122,65,0.14), rgba(33,97,140,0.10));
      border: 1px solid rgba(16,34,43,0.08);
      margin-bottom: 1rem;
    }
    .metric-card {
      padding: 0.75rem 1rem;
      border-radius: 16px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(16,34,43,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

snapshot = _load_snapshot()
leaderboard = _load_leaderboard()
_snapshot_leaderboard_rankings(leaderboard)
registry = snapshot.get("registry", {}) if isinstance(snapshot, dict) else {}
regime = snapshot.get("regime", {}) if isinstance(snapshot, dict) else {}
services = snapshot.get("services", {}) if isinstance(snapshot, dict) else {}
collector_status = snapshot.get("collector_status", {}) if isinstance(snapshot, dict) else {}
soak_summary = snapshot.get("soak_summary", {}) if isinstance(snapshot, dict) else {}
model_scheduler_status = snapshot.get("model_scheduler_status", {}) if isinstance(snapshot, dict) else {}
routing_policy = snapshot.get("routing_policy", {}) if isinstance(snapshot, dict) else {}

st.markdown(
    f"""
    <div class="hero">
      <h2 style="margin:0;">QDev Control Plane</h2>
      <p style="margin:0.35rem 0 0 0;">
        Current regime: <strong>{regime.get('regime', 'unknown')}</strong>
        &nbsp;|&nbsp; Active pipeline: <strong>{registry.get('active_pipeline_id', '-')}</strong>
        &nbsp;|&nbsp; Champion: <strong>{registry.get('champion_pipeline_id', '-')}</strong>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Actions")
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    refresh_seconds = st.selectbox("Refresh Interval (sec)", options=[15, 30, 60, 300], index=1)
    if auto_refresh:
        st.markdown(f"<meta http-equiv='refresh' content='{int(refresh_seconds)}'>", unsafe_allow_html=True)

    if st.button("Refresh Snapshot", use_container_width=True):
        code, output = _refresh_snapshot()
        if code == 0:
            st.success("Snapshot refreshed")
        else:
            st.error(output or "Refresh failed")
        st.rerun()

    if st.button("Run Model Cycle Now", use_container_width=True):
        code, output = _run_command(
            [_python_executable(), str(ROOT / "scripts" / "run_model_scheduler.py"), "--iterations", "1", "--suite", "shortlist", "--refresh-dataset"]
        )
        if code == 0:
            st.success("Model cycle completed")
        else:
            st.error(output or "Model cycle failed")
        st.rerun()

    st.caption("Services")
    for service_name in ("collector", "model_scheduler", "paper_soak"):
        cols = st.columns(2)
        if cols[0].button(f"Start {service_name}", key=f"start_{service_name}", use_container_width=True):
            code, output = _run_command([_python_executable(), str(ROOT / "scripts" / "manage_runtime_service.py"), service_name, "start"])
            if code == 0:
                st.success(f"{service_name} started")
            else:
                st.error(output or f"Failed to start {service_name}")
            st.rerun()
        if cols[1].button(f"Stop {service_name}", key=f"stop_{service_name}", use_container_width=True):
            code, output = _run_command([_python_executable(), str(ROOT / "scripts" / "manage_runtime_service.py"), service_name, "stop"])
            if code == 0:
                st.info(f"{service_name} stopped")
            else:
                st.error(output or f"Failed to stop {service_name}")
            st.rerun()

    st.caption("Champion Override")
    override_options = leaderboard["pipeline_id"].tolist() if not leaderboard.empty else []
    selected_pipeline = st.selectbox("Manual champion", options=override_options or ["-"])
    if st.button("Apply Override", use_container_width=True, disabled=not override_options):
        code, output = _run_command([_python_executable(), str(ROOT / "scripts" / "set_champion.py"), "--pipeline-id", selected_pipeline])
        if code == 0:
            st.success(f"Manual champion set to {selected_pipeline}")
        else:
            st.error(output or "Failed to set champion")
        st.rerun()
    if st.button("Clear Override", use_container_width=True):
        code, output = _run_command([_python_executable(), str(ROOT / "scripts" / "set_champion.py"), "--clear"])
        if code == 0:
            st.info("Champion override cleared")
        else:
            st.error(output or "Failed to clear override")
        st.rerun()

top_cols = st.columns(5)
dataset = snapshot.get("dataset", {}) if isinstance(snapshot, dict) else {}
latest_soak = soak_summary.get("latest_status", {}) if isinstance(soak_summary, dict) else {}
top_cols[0].metric("Regime", str(regime.get("regime", "unknown")))
top_cols[1].metric("Active Pipeline", str(registry.get("active_pipeline_id", "-")))
top_cols[2].metric("Champion", str(registry.get("champion_pipeline_id", "-")))
top_cols[3].metric("Dataset Version", str(dataset.get("version", "-")))
top_cols[4].metric("Soak Health", _status_badge(bool(latest_soak.get("healthy", False))))

service_rows = []
for name, payload in services.items():
    if isinstance(payload, dict):
        service_rows.append(
            {
                "service": name,
                "running": bool(payload.get("running", False)),
                "pid": payload.get("pid"),
                "started_at": payload.get("started_at"),
                "log_path": payload.get("log_path"),
            }
        )

left, right = st.columns([1.3, 1.0])

with left:
    st.subheader("Leaderboard")
    if leaderboard.empty:
        st.info("Leaderboard not available yet.")
    else:
        _MACRO_DAILY_FAMILIES = {"Deep Learning", "Financial + DL", "Bayesian", "SDE", "Factor Model"}
        _INTRADAY_FAMILIES = {"Intraday", "Intraday Swing"}
        _HFT_FAMILIES = {"HFT", "HFT Microstructure", "OnlineSGD"}

        def _slice_horizon(df: pd.DataFrame, families: set[str]) -> pd.DataFrame:
            if "family" not in df.columns:
                return pd.DataFrame()
            return df[df["family"].isin(families)].copy()

        _lb_macro = _slice_horizon(leaderboard, _MACRO_DAILY_FAMILIES)
        _lb_intraday = _slice_horizon(leaderboard, _INTRADAY_FAMILIES)
        _lb_hft = _slice_horizon(leaderboard, _HFT_FAMILIES)

        _lb_tabs = st.tabs(["Macro / Daily", "Intraday / Swing", "HFT Microstructure"])

        def _render_leaderboard_tab(tab_df: pd.DataFrame, title: str) -> None:
            if tab_df.empty:
                st.info(f"No {title} models in leaderboard yet.")
                return
            _frame = tab_df.head(12).copy()
            _needed_cols = {"pipeline_id", "family", "decision", "final_score",
                            "test_summary.total_return_pct", "test_summary.sharpe_ratio",
                            "test_summary.max_drawdown_pct"}
            if _needed_cols.issubset(_frame.columns):
                _bar = px.bar(
                    _frame,
                    x="pipeline_id",
                    y="final_score",
                    color="family",
                    hover_data=["decision", "test_summary.total_return_pct", "test_summary.sharpe_ratio"],
                    title=f"{title} — Pipeline Final Score",
                )
                st.plotly_chart(_bar, use_container_width=True)
                _frame["_sharpe_size"] = _frame["test_summary.sharpe_ratio"].clip(lower=0)
                _scatter = px.scatter(
                    _frame,
                    x="test_summary.max_drawdown_pct",
                    y="test_summary.total_return_pct",
                    size="_sharpe_size",
                    color="family",
                    hover_name="pipeline_id",
                    hover_data=["test_summary.sharpe_ratio"],
                    title=f"{title} — Return vs Drawdown",
                )
                st.plotly_chart(_scatter, use_container_width=True)
                st.dataframe(
                    _frame[[
                        "pipeline_id", "family", "decision", "final_score",
                        "test_summary.total_return_pct", "test_summary.sharpe_ratio",
                        "test_summary.max_drawdown_pct",
                    ]],
                    use_container_width=True,
                )
            else:
                st.dataframe(_frame, use_container_width=True)

        with _lb_tabs[0]:
            _render_leaderboard_tab(_lb_macro, "Macro / Daily")

        with _lb_tabs[1]:
            _render_leaderboard_tab(_lb_intraday, "Intraday / Swing")

        with _lb_tabs[2]:
            if _lb_hft.empty:
                st.info("No HFT Microstructure models in leaderboard yet.")
            else:
                _hft_display = _lb_hft.copy()
                _hft_cols_extra = [c for c in ["hft.hit_rate_pct", "hft.mae_bps", "hft.tick_count", "hft.avg_spread_bps"] if c in _hft_display.columns]
                _bar_hft = px.bar(
                    _hft_display,
                    x="pipeline_id",
                    y="final_score",
                    color="pipeline_id",
                    hover_data=["notes"] + _hft_cols_extra,
                    title="HFT Microstructure — Final Score",
                )
                st.plotly_chart(_bar_hft, use_container_width=True)
                _base_cols = ["pipeline_id", "decision", "final_score",
                              "test_summary.total_return_pct", "test_summary.sharpe_ratio",
                              "test_summary.max_drawdown_pct"]
                _show_cols = [c for c in _base_cols + _hft_cols_extra if c in _hft_display.columns]
                st.dataframe(_hft_display[_show_cols], use_container_width=True)

with right:
    st.subheader("Runtime Status")
    if service_rows:
        st.dataframe(pd.DataFrame(service_rows), use_container_width=True)
    else:
        st.info("No service metadata recorded yet.")

    st.subheader("Model Cycle")
    if model_scheduler_status:
        st.json(
            {
                "suite": model_scheduler_status.get("suite"),
                "healthy": model_scheduler_status.get("healthy"),
                "last_cycle_started_at": model_scheduler_status.get("last_cycle_started_at"),
                "last_cycle_finished_at": model_scheduler_status.get("last_cycle_finished_at"),
                "latest_run": model_scheduler_status.get("latest_run"),
                "report_path": model_scheduler_status.get("report_path"),
            }
        )
    else:
        st.info("Model scheduler status unavailable.")

    st.subheader("Current Regime Metrics")
    metrics = regime.get("metrics", {}) if isinstance(regime, dict) else {}
    if metrics:
        st.json(metrics)
    else:
        st.info("Regime metrics unavailable.")

st.subheader("Ranking History")
ranking_history = _load_leaderboard_history()
if ranking_history.empty:
    st.info("Ranking history will appear here once more snapshots are collected (auto-refreshes over time).")
else:
    top_n = 10
    top_pipelines = (
        ranking_history[ranking_history["timestamp"] == ranking_history["timestamp"].max()]
        .nsmallest(top_n, "score_rank")["pipeline_id"]
        .tolist()
    )
    filtered_history = ranking_history[ranking_history["pipeline_id"].isin(top_pipelines)]

    rh_col1, rh_col2, rh_col3 = st.columns(3)

    with rh_col1:
        fig_return = px.line(
            filtered_history.sort_values("timestamp"),
            x="timestamp",
            y="return_rank",
            color="pipeline_id",
            title="Total Return (%) Rank over Time",
            labels={"return_rank": "Rank (1=best)", "timestamp": "Time"},
        )
        fig_return.update_yaxes(autorange="reversed")
        fig_return.update_layout(legend=dict(font=dict(size=9)), margin=dict(t=40, b=20))
        st.plotly_chart(fig_return, use_container_width=True)

    with rh_col2:
        fig_sharpe = px.line(
            filtered_history.sort_values("timestamp"),
            x="timestamp",
            y="sharpe_rank",
            color="pipeline_id",
            title="Sharpe Ratio Rank over Time",
            labels={"sharpe_rank": "Rank (1=best)", "timestamp": "Time"},
        )
        fig_sharpe.update_yaxes(autorange="reversed")
        fig_sharpe.update_layout(legend=dict(font=dict(size=9)), margin=dict(t=40, b=20))
        st.plotly_chart(fig_sharpe, use_container_width=True)

    with rh_col3:
        fig_score = px.line(
            filtered_history.sort_values("timestamp"),
            x="timestamp",
            y="score_rank",
            color="pipeline_id",
            title="Final Score Rank over Time",
            labels={"score_rank": "Rank (1=best)", "timestamp": "Time"},
        )
        fig_score.update_yaxes(autorange="reversed")
        fig_score.update_layout(legend=dict(font=dict(size=9)), margin=dict(t=40, b=20))
        st.plotly_chart(fig_score, use_container_width=True)

st.subheader("Live HFT Monitor")
_hft_status = _load_hft_status()
if _hft_status is None:
    st.info("HFT engine not running or no data yet.")
else:
    import plotly.graph_objects as go

    _hft_symbols_data: dict = _hft_status.get("symbols", {})
    _hft_model: dict = _hft_status.get("model", {})
    _hft_broker: dict = _hft_status.get("broker", {})
    _hft_updated_at: str = _hft_status.get("updated_at", "")

    st.caption(f"Last updated: {_hft_updated_at}  |  Engine tick: {_hft_status.get('tick_counter', 0):,}")

    hft_col1, hft_col2, hft_col3 = st.columns(3)

    with hft_col1:
        st.markdown("**Market State & Price**")
        _state_colors = {
            "STABLE_TREND": "green",
            "VOLATILE_TREND": "orange",
            "RANGING": "blue",
            "HIGH_TOXICITY": "red",
            "MEAN_REVERTING": "purple",
            "CRISIS": "red",
            "LOW_ACTIVITY": "gray",
            "UNKNOWN": "gray",
        }
        for _sym, _sdata in _hft_symbols_data.items():
            st.markdown(f"**{_sym}**")
            _state_name = _sdata.get("market_state", "UNKNOWN")
            _state_color = _state_colors.get(_state_name, "gray")
            st.markdown(
                f'<span style="background:{_state_color};color:white;padding:2px 8px;border-radius:8px;font-size:0.8em;">{_state_name}</span>',
                unsafe_allow_html=True,
            )
            _mc1, _mc2 = st.columns(2)
            _mc1.metric("Price", f"{_sdata.get('price', 0.0):,.2f}")
            _mc2.metric("Spread", f"{_sdata.get('spread', 0.0):.4f}")
            _mc3, _mc4 = st.columns(2)
            _mc3.metric("Bid", f"{_sdata.get('bid', 0.0):,.2f}")
            _mc4.metric("Ask", f"{_sdata.get('ask', 0.0):,.2f}")
            _obi_val = float(_sdata.get("obi", 0.0))
            st.progress(min(max((_obi_val + 1.0) / 2.0, 0.0), 1.0), text=f"OBI: {_obi_val:+.3f}")
            st.divider()

    with hft_col2:
        st.markdown("**Microstructure Features**")
        _feature_names = ["toxicity_vpin", "volatility_burst", "intensity"]
        _fig_micro = go.Figure()
        for _sym, _sdata in _hft_symbols_data.items():
            _vals = [float(_sdata.get(f, 0.0)) for f in _feature_names]
            _fig_micro.add_trace(go.Bar(name=_sym, x=_feature_names, y=_vals))
        _fig_micro.update_layout(
            barmode="group",
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(_fig_micro, use_container_width=True)

        st.markdown("**Microprice**")
        for _sym, _sdata in _hft_symbols_data.items():
            st.metric(f"{_sym} Microprice", f"{_sdata.get('microprice', 0.0):,.4f}")

    with hft_col3:
        st.markdown("**Online Learning**")
        st.metric("Total SGD Updates", f"{_hft_model.get('total_updates', 0):,}")
        st.metric("Feature Dimensions", _hft_model.get("n_features", 5))

        _hft_ticks_df = _load_hft_ticks(limit=200)
        if not _hft_ticks_df.empty and "prediction_bps" in _hft_ticks_df.columns and "symbol" in _hft_ticks_df.columns:
            _fig_pred = px.line(
                _hft_ticks_df.sort_values("timestamp"),
                x="timestamp",
                y="prediction_bps",
                color="symbol",
                title="Prediction (bps) over Time",
                labels={"prediction_bps": "bps", "timestamp": "Time"},
            )
            _fig_pred.update_layout(margin=dict(t=30, b=10), height=220, legend=dict(font=dict(size=9)))
            st.plotly_chart(_fig_pred, use_container_width=True)

        if not _hft_ticks_df.empty and "target_weight" in _hft_ticks_df.columns and "symbol" in _hft_ticks_df.columns:
            _fig_wt = px.line(
                _hft_ticks_df.sort_values("timestamp"),
                x="timestamp",
                y="target_weight",
                color="symbol",
                title="Target Weight over Time",
                labels={"target_weight": "Weight", "timestamp": "Time"},
            )
            _fig_wt.update_layout(margin=dict(t=30, b=10), height=220, legend=dict(font=dict(size=9)))
            st.plotly_chart(_fig_wt, use_container_width=True)

    hft_bot_left, hft_bot_right = st.columns(2)

    with hft_bot_left:
        st.markdown("**Broker PnL over Time**")
        _hft_ticks_df2 = _load_hft_ticks(limit=200) if "_hft_ticks_df" not in dir() else _hft_ticks_df
        if not _hft_ticks_df2.empty and "pnl" in _hft_ticks_df2.columns:
            _fig_pnl = px.line(
                _hft_ticks_df2.sort_values("timestamp"),
                x="timestamp",
                y="pnl",
                title="Broker PnL",
                labels={"pnl": "PnL", "timestamp": "Time"},
            )
            _fig_pnl.update_layout(margin=dict(t=30, b=10), height=260)
            st.plotly_chart(_fig_pnl, use_container_width=True)
        else:
            st.info("PnL history not available yet.")
        _bc1, _bc2, _bc3, _bc4 = st.columns(4)
        _bc1.metric("Cash", f"{_hft_broker.get('cash', 0.0):,.2f}")
        _bc2.metric("Inventory", f"{_hft_broker.get('inventory', 0.0):.4f}")
        _bc3.metric("PnL", f"{_hft_broker.get('pnl', 0.0):+.2f}")
        _bc4.metric("Active Orders", _hft_broker.get("active_orders", 0))

    with hft_bot_right:
        st.markdown("**Latest Tick Data (last 20)**")
        _hft_ticks_tail = _load_hft_ticks(limit=200)
        if not _hft_ticks_tail.empty:
            _display_cols = [c for c in ["timestamp", "symbol", "price", "spread", "obi", "market_state", "prediction_bps", "target_weight"] if c in _hft_ticks_tail.columns]
            st.dataframe(_hft_ticks_tail.tail(20)[_display_cols], use_container_width=True)
        else:
            st.info("No tick records available yet.")

st.subheader("3-Layer Routing Policy")
if routing_policy:
    _rp_layers = routing_policy.get("layers", {})
    _rp_regime = routing_policy.get("regime", "unknown")
    _rp_ts = routing_policy.get("timestamp", "")

    st.caption(f"Generated: {_rp_ts}  |  Regime: **{_rp_regime}**")

    _layer_cols = st.columns(3)

    def _layer_badge(active: bool) -> str:
        return "🟢 ACTIVE" if active else "🔴 OFF"

    with _layer_cols[0]:
        _m = _rp_layers.get("macro_daily", {})
        st.markdown(f"**Layer 1 — Macro / Daily** {_layer_badge(_m.get('active', False))}")
        st.metric("Champion", _m.get("champion_pipeline_id", "-"))
        st.caption(f"Regime hint: {_m.get('regime_hint', '-')}  |  Alloc scale: {_m.get('allocation_scale', 0):.0%}")
        _challengers_m = _m.get("challenger_ids", [])
        if _challengers_m:
            st.caption(f"Challengers: {', '.join(_challengers_m)}")

    with _layer_cols[1]:
        _i = _rp_layers.get("intraday_swing", {})
        st.markdown(f"**Layer 2 — Intraday / Swing** {_layer_badge(_i.get('active', False))}")
        st.metric("Champion", _i.get("champion_pipeline_id") or "—")
        st.caption(_i.get("reason", "-"))

    with _layer_cols[2]:
        _h = _rp_layers.get("hft", {})
        st.markdown(f"**Layer 3 — HFT** {_layer_badge(_h.get('active', False))}")
        st.metric("Champion", _h.get("champion_pipeline_id") or "—")
        st.caption(f"allow_hft: {_h.get('allow_hft')}  |  {_h.get('reason', '-')}")
        _challengers_h = _h.get("challenger_ids", [])
        if _challengers_h:
            st.caption(f"Challengers: {', '.join(_challengers_h)}")

    # Mandatory overlays
    _overlays = routing_policy.get("mandatory_overlays", {})
    if _overlays:
        with st.expander("Mandatory Overlays", expanded=False):
            for k, v in _overlays.items():
                st.markdown(f"**{k}** — {v.get('description', '')}  `{'ON' if v.get('active') else 'OFF'}`")
else:
    st.info("Routing policy not yet generated — run model cycle to populate.")

st.subheader("Regime Routing Table")
assignments = registry.get("regime_assignments", {}) if isinstance(registry, dict) else {}
assignment_rows = []
for regime_name, payload in assignments.items():
    if isinstance(payload, dict):
        assignment_rows.append(
            {
                "regime": regime_name,
                "pipeline_id": payload.get("pipeline_id"),
                "family": payload.get("family"),
                "final_score": payload.get("final_score"),
                "regime_score": payload.get("regime_score"),
            }
        )
if assignment_rows:
    st.dataframe(pd.DataFrame(assignment_rows), use_container_width=True)

st.subheader("Collector")
symbol_rows = collector_status.get("symbols", []) if isinstance(collector_status, dict) else []
if symbol_rows:
    st.dataframe(pd.DataFrame(symbol_rows), use_container_width=True)
else:
    st.info("Collector has not written a status file yet.")

st.subheader("Soak Timeline")
timeline = soak_summary.get("timeline", []) if isinstance(soak_summary, dict) else []
if timeline:
    timeline_frame = pd.DataFrame(timeline)
    if "recorded_at" in timeline_frame.columns:
        timeline_frame["recorded_at"] = pd.to_datetime(timeline_frame["recorded_at"])
    timeline_frame["healthy_numeric"] = timeline_frame["healthy"].astype(int)
    soak_chart = px.line(
        timeline_frame,
        x="recorded_at",
        y=["healthy_numeric", "stream_age_seconds"],
        title="Soak Health and Stream Age",
    )
    st.plotly_chart(soak_chart, use_container_width=True)
    st.dataframe(timeline_frame.tail(20), use_container_width=True)
else:
    st.info("No soak timeline records available.")

st.subheader("Recent Logs")
log_tabs = st.tabs(["Collector Log", "Model Scheduler Log", "Collector JSONL", "Scheduler JSONL", "Soak Records"])

with log_tabs[0]:
    collector_log = _load_text_tail(LOG_ROOT / "collector.log")
    if collector_log:
        st.code(collector_log, language="text")
    else:
        st.info("Collector log not available.")

with log_tabs[1]:
    scheduler_log = _load_text_tail(LOG_ROOT / "model_scheduler.log")
    if scheduler_log:
        st.code(scheduler_log, language="text")
    else:
        st.info("Model scheduler log not available.")

with log_tabs[2]:
    collector_jsonl = _load_jsonl_tail(LOG_ROOT / "data_collector.jsonl")
    if collector_jsonl.empty:
        st.info("Collector JSONL log not available.")
    else:
        st.dataframe(collector_jsonl, use_container_width=True)

with log_tabs[3]:
    scheduler_jsonl = _load_jsonl_tail(LOG_ROOT / "model_scheduler.jsonl")
    if scheduler_jsonl.empty:
        st.info("Model scheduler JSONL log not available.")
    else:
        st.dataframe(scheduler_jsonl, use_container_width=True)

with log_tabs[4]:
    soak_jsonl = _load_jsonl_tail(SOAK_RECORDS_PATH)
    if soak_jsonl.empty:
        st.info("Soak records are not available yet.")
    else:
        st.dataframe(soak_jsonl, use_container_width=True)
