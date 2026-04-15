from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DOC = ROOT / "docs" / "md" / "알고리즘_파이프라인_후보_라이브러리.md"
OUTPUT_DIR = ROOT / "docs" / "pipeline_library"
README_PATH = OUTPUT_DIR / "README.md"
REFERENCE_MANIFEST = ROOT / "docs" / "references" / "reference_manifest.json"


@dataclass
class PipelineEntry:
    pipeline_id: str
    name: str
    model: str = ""
    input_bundle: str = ""
    target: str = ""
    execution: str = ""
    advantages: str = ""
    weakness: str = ""
    difficulty: str = ""
    fit: str = ""
    priority: str = ""
    raw_lines: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        direct = {"E001", "E002", "E003", "F001", "F023"}
        partial = {"B007", "B014", "F002", "F009", "F012", "F015", "F019", "F025", "F028", "F030", "F031", "F035", "F037", "F039", "F041", "F042", "F043", "F045"}
        if self.pipeline_id in direct:
            return "direct code path"
        if self.pipeline_id in partial:
            return "supporting components implemented"
        return "research candidate"


@dataclass
class ParsedInventory:
    families: dict[str, dict[str, object]]
    shortlist_primary: list[tuple[str, str]]
    shortlist_secondary: list[tuple[str, str]]
    rejection_criteria: list[str]
    implementation_steps: list[str]


FAMILY_DOCS = {
    "A": ("01_bayesian_pipelines.md", "Bayesian Pipelines"),
    "B": ("02_sde_pipelines.md", "SDE Pipelines"),
    "C": ("03_rl_pipelines.md", "RL Pipelines"),
    "D": ("04_financial_dl_pipelines.md", "Financial + DL Pipelines"),
}

FAMILY_REFERENCE_TAGS = {
    "01_bayesian_pipelines.md": {"bayesian", "regime", "monitoring", "macro"},
    "02_sde_pipelines.md": {"sde", "mean_reversion", "volatility", "jump", "execution"},
    "03_rl_pipelines.md": {"rl", "portfolio"},
    "04_financial_dl_pipelines.md": {"financial_dl", "tabular", "time_series"},
    "05_priority_shortlist.md": {"bayesian", "sde", "rl", "financial_dl", "portfolio", "regime", "mean_reversion", "volatility", "execution"},
    "06_rejected_or_archived_pipelines.md": {"bayesian", "sde", "rl", "financial_dl"},
}

IMPLEMENTATION_NOTES = {
    "01_bayesian_pipelines.md": [
        "현재 코드는 Bayesian sampler나 particle filter를 풀로 구현하지 않았고, HMM 기반 regime detector와 risk gate가 Bayesian-style overlay를 대신한다.",
        "즉시 실행 가능한 Bayesian-like 경로는 `SimpleHMMRegimeDetector`와 `RiskManager`를 통해 B007/B014 계열을 부분 검증하는 수준이다.",
    ],
    "02_sde_pipelines.md": [
        "전용 SDE solver는 아직 없고, mean-reversion / volatility overlay 후보를 연구용으로 분류한다.",
        "실행 레이어는 `BacktestEngine`, `RiskManager`, 그리고 HFT replay 기반 synthetic validation이 담당한다.",
    ],
    "03_rl_pipelines.md": [
        "이 레포에는 RL trainer나 environment가 없으므로, RL 후보는 설계 문서와 실험 계획을 위한 inventory로 유지한다.",
        "paper/live safety gate와 health loop가 RL 산출물을 올릴 때의 운영 가드레일 역할을 한다.",
    ],
    "04_financial_dl_pipelines.md": [
        "`build_technical_features`, `SimpleHMMRegimeDetector`, `LightGBMRanker`, `MLStrategy`, `BacktestEngine`가 F001/F023/F031류 baseline을 직접 지원한다.",
        "텍스트, graph, transformer, TFT 계열은 문서화는 되었지만 아직 별도 모델 구현이 필요하다.",
    ],
    "05_priority_shortlist.md": [
        "이 문서는 1차 구현 우선순위와 2차 연구 우선순위를 함께 보관한다.",
        "현재 레포에서는 F001/F023/F031과 HFT replay path가 가장 직접적인 executable baseline이다.",
    ],
    "06_rejected_or_archived_pipelines.md": [
        "이 문서는 명시적으로 폐기된 pipeline ID를 나열하기보다, 미래에 drop해야 할 기준을 보관한다.",
        "point-in-time, cost, execution, maintenance 조건이 실패하면 archived 상태로 이동한다.",
    ],
}

DIRECT_IMPL_IDS = {"E001", "E002", "E003", "F001", "F023"}
PARTIAL_IMPL_IDS = {"B007", "B014", "F002", "F009", "F012", "F015", "F019", "F025", "F028", "F030", "F031", "F035", "F037", "F039", "F041", "F042", "F043", "F045"}


def _escape_cell(value: str) -> str:
    return value.replace("\n", "<br>").replace("|", r"\|").strip()


def _extract_field(lines: Iterable[str], labels: Iterable[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            continue
        for label in labels:
            match = re.match(rf"^-\s*{re.escape(label)}\s*:\s*(.+)$", stripped)
            if match:
                return match.group(1).strip()
    return ""


def parse_inventory(text: str) -> ParsedInventory:
    families: dict[str, dict[str, object]] = {}
    current_family: str | None = None
    current_entry: PipelineEntry | None = None

    family_heading = re.compile(r"^#\s+([A-D])\.\s+(.+?)\s+파이프라인$")
    pipeline_heading = re.compile(r"^##\s+([A-Z]\d{3})\.\s+(.+)$")

    for line in text.splitlines():
        family_match = family_heading.match(line)
        if family_match:
            current_family = family_match.group(1)
            families.setdefault(
                current_family,
                {"title": family_match.group(2), "entries": []},
            )
            current_entry = None
            continue

        pipeline_match = pipeline_heading.match(line)
        if pipeline_match and current_family:
            current_entry = PipelineEntry(
                pipeline_id=pipeline_match.group(1),
                name=pipeline_match.group(2).strip(),
            )
            families[current_family]["entries"].append(current_entry)
            continue

        if current_entry is not None:
            current_entry.raw_lines.append(line)

    for family in families.values():
        entries: list[PipelineEntry] = family["entries"]  # type: ignore[assignment]
        for entry in entries:
            lines = [line.strip() for line in entry.raw_lines if line.strip()]
            entry.model = _extract_field(lines, ["모델"])
            entry.input_bundle = _extract_field(lines, ["입력"])
            entry.target = _extract_field(lines, ["타깃"])
            entry.execution = _extract_field(lines, ["실행"])
            entry.advantages = _extract_field(lines, ["장점"])
            entry.weakness = _extract_field(lines, ["약점", "한계"])
            entry.difficulty = _extract_field(lines, ["난이도"])
            entry.fit = _extract_field(lines, ["실전 적합성"])
            entry.priority = _extract_field(lines, ["우선순위"])

    shortlist_primary = _parse_shortlist_section(text, "## 5.1", "## 5.2")
    shortlist_secondary = _parse_shortlist_section(text, "## 5.2", "# 6.")
    rejection_criteria = _parse_bullet_list(_extract_section(text, "# 10. 탈락 기준", "# 11. 결론"))
    implementation_steps = _parse_bullet_list(_extract_section(text, "# 9. 추천 구현 순서", "# 10. 탈락 기준"))
    return ParsedInventory(families, shortlist_primary, shortlist_secondary, rejection_criteria, implementation_steps)


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        end = len(text)
    return text[start:end]


def _parse_bullet_list(section: str) -> list[str]:
    items: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _parse_shortlist_section(text: str, start_marker: str, end_marker: str) -> list[tuple[str, str]]:
    section = _extract_section(text, start_marker, end_marker)
    parsed: list[tuple[str, str]] = []
    for item in _parse_bullet_list(section):
        match = re.match(r"^([A-Z]\d{3})\s+(.+)$", item)
        if match:
            parsed.append((match.group(1), match.group(2).strip()))
    return parsed


def load_reference_manifest() -> list[dict[str, object]]:
    if not REFERENCE_MANIFEST.exists():
        return []
    with REFERENCE_MANIFEST.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_references(doc_name: str, manifest: list[dict[str, object]]) -> list[dict[str, object]]:
    if not manifest:
        return []
    if doc_name == "00_pipeline_catalog.md":
        return manifest
    allowed_tags = FAMILY_REFERENCE_TAGS.get(doc_name, set())
    if not allowed_tags:
        return manifest
    return [record for record in manifest if allowed_tags.intersection(set(record.get("tags", [])))]


def render_reference_section(doc_name: str, references: list[dict[str, object]]) -> str:
    if not references:
        return (
            "## Reference Anchors\n\n"
            "- Reference bundle not yet downloaded. Run `python scripts/download_pipeline_references.py` first.\n"
        )

    lines = ["## Reference Anchors", ""]
    for record in references:
        filename = record.get("filename", "")
        local_link = f"../references/papers/{filename}" if filename else record.get("source_url", "")
        title = record.get("title", record.get("id", "reference"))
        tags = ", ".join(record.get("tags", []))
        lines.append(f"- [{title}]({local_link}) - {tags}")
    lines.append("")
    return "\n".join(lines)


def render_inventory_table(entries: list[PipelineEntry]) -> str:
    header = (
        "| ID | Name | Model | Input Bundle | Target | Execution / Overlay | Difficulty | Fit | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    )
    rows = []
    for entry in entries:
        rows.append(
            "| {id} | {name} | {model} | {input_bundle} | {target} | {execution} | {difficulty} | {fit} | {status} |".format(
                id=_escape_cell(entry.pipeline_id),
                name=_escape_cell(entry.name),
                model=_escape_cell(entry.model),
                input_bundle=_escape_cell(entry.input_bundle),
                target=_escape_cell(entry.target),
                execution=_escape_cell(entry.execution),
                difficulty=_escape_cell(entry.difficulty),
                fit=_escape_cell(entry.fit),
                status=_escape_cell(entry.status),
            )
        )
    return "\n".join((*header, *rows))


def render_priority_table(items: list[tuple[str, str]], stage_label: str) -> str:
    lines = [
        "| ID | Name | Family | Repo Status |",
        "| --- | --- | --- | --- |",
    ]
    for pipeline_id, name in items:
        family = pipeline_id[0]
        status = "direct code path" if pipeline_id in DIRECT_IMPL_IDS else "supporting components implemented" if pipeline_id in PARTIAL_IMPL_IDS else "research candidate"
        lines.append(f"| {pipeline_id} | {_escape_cell(name)} | {family} | {status} |")
    return "\n".join(lines)


def render_family_doc(doc_name: str, family_letter: str, family_title: str, entries: list[PipelineEntry], references: list[dict[str, object]]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        f"# {family_title}",
        "",
        "<!-- generated by scripts/generate_pipeline_docs.py -->",
        "",
        f"- source: `docs/md/알고리즘_파이프라인_후보_라이브러리.md`",
        f"- generated_at: `{generated_at}`",
        f"- pipeline_count: `{len(entries)}`",
        f"- status_summary: `direct={sum(1 for entry in entries if entry.status == 'direct code path')}, partial={sum(1 for entry in entries if entry.status == 'supporting components implemented')}, research={sum(1 for entry in entries if entry.status == 'research candidate')}`",
        "",
        render_reference_section(doc_name, references),
        "## Repo Coverage",
        "",
    ]
    lines.extend(
        [
            "- `src/features/builder.py` provides the technical feature baseline for the LightGBM path.",
            "- `src/models/hmm.py` and `src/risk/manager.py` provide the regime/risk overlay used by the executable baseline.",
            "- `src/brokers/paper.py`, `src/monitoring/health.py`, and `src/ingestion/websocket_client.py` now cover the broker + health + replay operational path.",
            "",
        ]
    )
    notes = IMPLEMENTATION_NOTES.get(doc_name, [])
    if notes:
        lines.append("## Implementation Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("## Inventory Table")
    lines.append("")
    lines.append(render_inventory_table(entries))
    lines.append("")
    return "\n".join(lines)


def render_catalog_doc(parsed: ParsedInventory, references: list[dict[str, object]]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    family_rows = []
    for family_letter in ["A", "B", "C", "D"]:
        title = parsed.families.get(family_letter, {}).get("title", "")
        doc_name, doc_title = FAMILY_DOCS[family_letter]
        entries: list[PipelineEntry] = parsed.families.get(family_letter, {}).get("entries", [])  # type: ignore[assignment]
        family_rows.append(
            f"| {family_letter} | {title} | [{doc_title}](./{doc_name}) | {len(entries)} | "
            f"{sum(1 for entry in entries if entry.status == 'direct code path')} | "
            f"{sum(1 for entry in entries if entry.status == 'supporting components implemented')} | "
            f"{sum(1 for entry in entries if entry.status == 'research candidate')} |"
        )

    core_paths = [
        ("Features", "src/features/builder.py"),
        ("Regime model", "src/models/hmm.py"),
        ("Tabular model", "src/models/lgbm.py"),
        ("Risk", "src/risk/manager.py"),
        ("Mock broker", "src/brokers/mock.py"),
        ("Paper broker", "src/brokers/paper.py"),
        ("Replay stream", "src/ingestion/websocket_client.py"),
        ("Health monitor", "src/monitoring/health.py"),
        ("Backtest", "src/backtest/engine.py"),
    ]

    lines = [
        "# Pipeline Catalog",
        "",
        "<!-- generated by scripts/generate_pipeline_docs.py -->",
        "",
        f"- source: `docs/md/알고리즘_파이프라인_후보_라이브러리.md`",
        f"- generated_at: `{generated_at}`",
        f"- reference_bundle: `docs/references/index.md`",
        "",
        "## Execution Coverage",
        "",
        "- install/bootstrap path: passing",
        "- offline smoke path: passing",
        "- backtest path: passing",
        "- synthetic HFT replay path: passing",
        "- paper broker adapter: implemented with fake-client validation, external broker not exercised",
        "- live broker path: still gated by explicit environment flags",
        "",
        render_reference_section("00_pipeline_catalog.md", references),
        "## Core Repo Paths",
        "",
        "| Area | File |",
        "| --- | --- |",
    ]
    lines.extend(f"| {area} | `{path}` |" for area, path in core_paths)
    lines.extend(
        [
            "",
            "## Family Coverage",
            "",
            "| Family | Title | Doc | Pipelines | Direct | Partial | Research |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            *family_rows,
            "",
            "## What This Means",
            "",
            "- The repository is executable in research/mock/offline mode.",
            "- The documentation inventory is now grounded by downloaded paper references and by the concrete code paths above.",
            "- Detailed primary candidate dossiers are published in `./07_candidate_dossiers.md` with a folder guide in `./candidates/README.md` and individual markdown files under `./candidates/`.",
            "- Paper/live trading still needs external credentialed validation before it can be called production-ready.",
            "",
        ]
    )
    return "\n".join(lines)


def render_library_readme_doc(parsed: ParsedInventory) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    total_pipelines = sum(len(family.get("entries", [])) for family in parsed.families.values())
    direct_count = sum(
        1
        for family in parsed.families.values()
        for entry in family.get("entries", [])
        if entry.status == "direct code path"
    )
    partial_count = sum(
        1
        for family in parsed.families.values()
        for entry in family.get("entries", [])
        if entry.status == "supporting components implemented"
    )
    research_count = sum(
        1
        for family in parsed.families.values()
        for entry in family.get("entries", [])
        if entry.status == "research candidate"
    )
    family_order = ["A", "B", "C", "D"]
    file_map = [
        ("README.md", "This file: entry point and navigation hub"),
        ("00_pipeline_catalog.md", "technical catalog and execution coverage"),
        ("01_bayesian_pipelines.md", "Bayesian family inventory"),
        ("02_sde_pipelines.md", "SDE family inventory"),
        ("03_rl_pipelines.md", "RL family inventory"),
        ("04_financial_dl_pipelines.md", "Financial + DL family inventory"),
        ("05_hft_pipelines.md", "supplemental HFT-specific catalog"),
        ("05_priority_shortlist.md", "priority shortlist and execution order"),
        ("06_rejected_or_archived_pipelines.md", "rejection and archival criteria"),
        ("07_candidate_dossiers.md", "detailed dossiers for the primary shortlist"),
        ("candidates/README.md", "folder-level dossier map"),
        ("candidates/manifest.json", "machine-readable dossier manifest"),
    ]

    lines = [
        "# Pipeline Library",
        "",
        "<!-- generated by scripts/generate_pipeline_docs.py -->",
        "",
        f"- source: `docs/md/알고리즘_파이프라인_후보_라이브러리.md`",
        f"- generated_at: `{generated_at}`",
        f"- total_pipelines: `{total_pipelines}`",
        f"- execution_summary: `direct={direct_count}, partial={partial_count}, research={research_count}`",
        "",
        "## Purpose",
        "",
        "- This directory is the navigation layer for the pipeline inventory.",
        "- The generated docs split the library by family, priority, archive criteria, and detailed candidate dossiers.",
        "- The executable baseline lives in the catalog plus the shortlist; the long-form dossiers live under `./candidates/`.",
        "- `05_hft_pipelines.md` is a supplemental HFT catalog that complements the generated A-D family docs.",
        "",
        "## Recommended Reading Order",
        "",
        "1. [`Pipeline Catalog`](./00_pipeline_catalog.md) for the full execution map.",
        "2. [`Priority Shortlist`](./05_priority_shortlist.md) for the implementation order.",
        "3. [`Candidate Dossier Index`](./07_candidate_dossiers.md) for the 20 detailed primary dossiers.",
        "4. [`Candidate Folder README`](./candidates/README.md) for direct file browsing.",
        "5. The family-specific inventories (`01`-`04`) plus the supplemental HFT catalog (`05`) if you need a deeper slice.",
        "",
        "## File Map",
        "",
        "| File | Role |",
        "| --- | --- |",
    ]
    lines.extend(f"| [`{filename}`](./{filename}) | {role} |" for filename, role in file_map)
    lines.extend(
        [
            "",
            "## Family Snapshot",
            "",
            "| Family | Title | Pipelines | Direct | Partial | Research |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for family_letter in family_order:
        family = parsed.families.get(family_letter, {})
        entries: list[PipelineEntry] = family.get("entries", [])  # type: ignore[assignment]
        lines.append(
            f"| {family_letter} | {family.get('title', '')} | {len(entries)} | "
            f"{sum(1 for entry in entries if entry.status == 'direct code path')} | "
            f"{sum(1 for entry in entries if entry.status == 'supporting components implemented')} | "
            f"{sum(1 for entry in entries if entry.status == 'research candidate')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `F001`, `F023`, and `F031` remain the most direct executable baselines in the repo.",
            "- `07_candidate_dossiers.md` and `./candidates/` are generated from the current shortlist and can be expanded later.",
            "- `05_hft_pipelines.md` is supplemental and is not counted in the 120-pipeline source inventory from the master spec.",
            "- Paper/live trading is still gated by explicit environment flags and external credential validation.",
            "",
        ]
    )
    return "\n".join(lines)


def render_shortlist_doc(parsed: ParsedInventory, references: list[dict[str, object]]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Priority Shortlist",
        "",
        "<!-- generated by scripts/generate_pipeline_docs.py -->",
        "",
        f"- generated_at: `{generated_at}`",
        "- scope: 1차 핵심 후보 + 2차 연구 우선 후보",
        "",
        render_reference_section("05_priority_shortlist.md", references),
        "## 1차 핵심 후보",
        "",
        render_priority_table(parsed.shortlist_primary, "1차"),
        "",
        "## 2차 연구 우선 후보",
        "",
        render_priority_table(parsed.shortlist_secondary, "2차"),
        "",
        "## Implementation Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in IMPLEMENTATION_NOTES["05_priority_shortlist.md"])
    lines.append("")
    lines.extend(
        [
        "## Detailed Dossiers",
        "",
        "- The 20 primary candidate dossiers live in `./candidates/`.",
        "- Start with [`Candidate Dossier Index`](./07_candidate_dossiers.md) for the curated entry points and reference anchors.",
        "- Use [`Candidate Folder README`](./candidates/README.md) if you want a file-level map of the same dossiers.",
        "",
    ]
    )
    return "\n".join(lines)


def render_rejected_doc(parsed: ParsedInventory, references: list[dict[str, object]]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Rejected or Archived Pipelines",
        "",
        "<!-- generated by scripts/generate_pipeline_docs.py -->",
        "",
        f"- generated_at: `{generated_at}`",
        "- purpose: explicit rejection/archival criteria and future drop rules",
        "",
        render_reference_section("06_rejected_or_archived_pipelines.md", references),
        "## Rejection Criteria",
        "",
    ]
    lines.extend(f"- {item}" for item in parsed.rejection_criteria)
    lines.extend(
        [
            "",
            "## Implementation Notes",
            "",
        ]
    )
    lines.extend(f"- {note}" for note in IMPLEMENTATION_NOTES["06_rejected_or_archived_pipelines.md"])
    lines.extend(
        [
            "",
            "## Archive Handling",
            "",
            "- The source inventory did not enumerate explicit archived pipeline IDs.",
            "- Future candidates that fail point-in-time availability, cost, maintenance, or execution practicality should be moved here instead of being silently dropped.",
            "",
        ]
    )
    return "\n".join(lines)


def write_generated(path: Path, content: str, force: bool = False) -> bool:
    marker = "<!-- generated by scripts/generate_pipeline_docs.py -->"
    if path.exists() and not force:
        existing = path.read_text(encoding="utf-8")
        if marker not in existing and not existing.strip().startswith("# Title"):
            return False
    path.write_text(content, encoding="utf-8")
    return True


def create_docs(force: bool = False) -> None:
    if not SOURCE_DOC.exists():
        raise FileNotFoundError(f"Missing source inventory: {SOURCE_DOC}")

    parsed = parse_inventory(SOURCE_DOC.read_text(encoding="utf-8"))
    references = load_reference_manifest()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    created = 0
    readme_content = render_library_readme_doc(parsed)
    if write_generated(README_PATH, readme_content, force=force):
        created += 1

    for family_letter, (doc_name, doc_title) in FAMILY_DOCS.items():
        family = parsed.families.get(family_letter, {"entries": []})
        entries: list[PipelineEntry] = family.get("entries", [])  # type: ignore[assignment]
        content = render_family_doc(doc_name, family_letter, doc_title, entries, select_references(doc_name, references))
        if write_generated(OUTPUT_DIR / doc_name, content, force=force):
            created += 1

    catalog_content = render_catalog_doc(parsed, select_references("00_pipeline_catalog.md", references))
    if write_generated(OUTPUT_DIR / "00_pipeline_catalog.md", catalog_content, force=force):
        created += 1

    shortlist_content = render_shortlist_doc(parsed, select_references("05_priority_shortlist.md", references))
    if write_generated(OUTPUT_DIR / "05_priority_shortlist.md", shortlist_content, force=force):
        created += 1

    rejected_content = render_rejected_doc(parsed, select_references("06_rejected_or_archived_pipelines.md", references))
    if write_generated(OUTPUT_DIR / "06_rejected_or_archived_pipelines.md", rejected_content, force=force):
        created += 1

    print(f"Generated or refreshed {created} pipeline documentation files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the pipeline inventory docs from the master spec.")
    parser.add_argument("--force", action="store_true", help="Overwrite generated docs even if they already exist.")
    args = parser.parse_args()
    create_docs(force=args.force)
