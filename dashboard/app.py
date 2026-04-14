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


def _status_badge(value: bool) -> str:
    return "Healthy" if value else "Attention"


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
registry = snapshot.get("registry", {}) if isinstance(snapshot, dict) else {}
regime = snapshot.get("regime", {}) if isinstance(snapshot, dict) else {}
services = snapshot.get("services", {}) if isinstance(snapshot, dict) else {}
collector_status = snapshot.get("collector_status", {}) if isinstance(snapshot, dict) else {}
soak_summary = snapshot.get("soak_summary", {}) if isinstance(snapshot, dict) else {}
model_scheduler_status = snapshot.get("model_scheduler_status", {}) if isinstance(snapshot, dict) else {}

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
        chart_frame = leaderboard.head(12).copy()
        bar = px.bar(
            chart_frame,
            x="pipeline_id",
            y="final_score",
            color="family",
            hover_data=["decision", "test_summary.total_return_pct", "test_summary.sharpe_ratio"],
            title="Pipeline Final Score",
        )
        st.plotly_chart(bar, use_container_width=True)

        scatter = px.scatter(
            chart_frame,
            x="test_summary.max_drawdown_pct",
            y="test_summary.total_return_pct",
            size="test_summary.sharpe_ratio",
            color="family",
            hover_name="pipeline_id",
            title="Return vs Drawdown",
        )
        st.plotly_chart(scatter, use_container_width=True)
        st.dataframe(
            chart_frame[
                [
                    "pipeline_id",
                    "family",
                    "decision",
                    "final_score",
                    "test_summary.total_return_pct",
                    "test_summary.sharpe_ratio",
                    "test_summary.max_drawdown_pct",
                ]
            ],
            use_container_width=True,
        )

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

st.subheader("Routing")
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
