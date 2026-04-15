from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "docs" / "references" / "quant_resource_bundle.md"
MANIFEST_PATH = ROOT / "docs" / "references" / "quant_resource_bundle_manifest.json"
CLEAN_MD_PATH = ROOT / "docs" / "references" / "quant_resource_bundle_clean.md"
DOWNLOAD_SCRIPT_PATH = ROOT / "scripts" / "download_quant_resource_bundle.sh"
ARTIFACT_ROOT = ROOT / "artifacts" / "quant_resource_bundle"

TOP_SECTION_PRIORITY = {
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
}

PAPER_GROUP_PRIORITY = {
    1: 2,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
}

KIND_ORDER = {
    "paper": 0,
    "repo": 1,
    "package": 2,
}


@dataclass(slots=True)
class Resource:
    kind: str
    source: str
    label: str
    destination: str
    priority: int
    section: str
    order: int
    raw_command: str
    aliases: list[str] = field(default_factory=list)

    def as_manifest_item(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source": self.source,
            "label": self.label,
            "destination": self.destination,
            "priority": self.priority,
            "section": self.section,
            "order": self.order,
            "aliases": self.aliases,
            "raw_command": self.raw_command,
        }


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._+-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-_.") or "item"


def package_destination(spec: str) -> str:
    return f"pypi/{slugify(spec)}"


def repo_destination(url: str) -> str:
    clean = url.strip()
    if clean.endswith(".git"):
        clean = clean[:-4]
    parts = clean.rstrip("/").split("/")
    if len(parts) < 2:
        return f"repos/{slugify(clean)}"
    owner = parts[-2]
    repo = parts[-1]
    return f"repos/{slugify(owner)}__{slugify(repo)}"


def paper_destination(filename: str) -> str:
    return f"papers/{filename}"


def paper_label_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^\d+_\d+_", "", stem)
    stem = stem.replace("_", " ").strip()
    return stem or stem


def parse_repo_url(line: str) -> str:
    return line.split("#", 1)[0].strip().split(maxsplit=2)[2].strip()


def parse_pip_spec(line: str) -> str:
    return line.split("#", 1)[0].strip()[len("pip install ") :].strip()


def parse_wget(line: str) -> tuple[str, str]:
    line = line.split("#", 1)[0].strip()
    m = re.match(r'wget\s+-O\s+"([^"]+)"\s+"([^"]+)"$', line)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"wget\s+-O\s+'([^']+)'\s+'([^']+)'$", line)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"wget\s+-O\s+(\S+)\s+(\S+)$", line)
    if m:
        return m.group(1).strip('"').strip("'"), m.group(2).strip('"').strip("'")
    raise ValueError(f"Could not parse wget line: {line}")


def first_pass_resources(text: str) -> tuple[list[Resource], list[str]]:
    resources: list[Resource] = []
    duplicate_notes: list[str] = []

    pre, post = text.split("## 8. 추가 다운로드 섹션", 1)

    current_section: int | None = None
    in_code = False
    order = 0
    seen: dict[tuple[str, str], Resource] = {}

    for raw_line in pre.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.match(r"##\s+(\d+)\.", line)
        if heading:
            current_section = int(heading.group(1))
            continue
        if line.startswith("```"):
            in_code = not in_code
            continue
        if not in_code or current_section is None:
            continue

        if line.startswith("pip install "):
            spec = parse_pip_spec(line)
            key = ("package", spec.lower())
            priority = TOP_SECTION_PRIORITY.get(current_section, 3)
            resource = Resource(
                kind="package",
                source=spec,
                label=spec,
                destination=package_destination(spec),
                priority=priority,
                section=f"section-{current_section}",
                order=order,
                raw_command=line,
            )
        elif line.startswith("git clone "):
            url = parse_repo_url(line)
            key = ("repo", url.rstrip("/").removesuffix(".git").lower())
            priority = TOP_SECTION_PRIORITY.get(current_section, 3)
            resource = Resource(
                kind="repo",
                source=url,
                label=url.rstrip("/").split("/")[-1].removesuffix(".git"),
                destination=repo_destination(url),
                priority=priority,
                section=f"section-{current_section}",
                order=order,
                raw_command=line,
            )
        else:
            continue

        existing = seen.get(key)
        if existing is not None:
            existing.aliases.append(resource.label)
            duplicate_notes.append(
                f"- removed duplicate {resource.kind}: {resource.source} (alias {resource.label})"
            )
            continue
        seen[key] = resource
        resources.append(resource)
        order += 1

    paper_block = re.search(r"```bash\s*(.*?)\s*```", post, flags=re.S)
    if paper_block is None:
        raise ValueError("Could not locate paper download code block in bundle.")

    current_group: int | None = None
    in_code = True
    for raw_line in paper_block.group(1).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        group_heading = re.match(r"#\s+(\d+)\.", line)
        if group_heading:
            current_group = int(group_heading.group(1))
            continue
        if not in_code or current_group is None:
            continue
        if not line.startswith("wget "):
            continue

        filename, url = parse_wget(line)
        key = ("paper", url.rstrip("/").lower())
        priority = PAPER_GROUP_PRIORITY.get(current_group, 2)
        resource = Resource(
            kind="paper",
            source=url,
            label=paper_label_from_filename(filename),
            destination=paper_destination(filename),
            priority=priority,
            section=f"paper-group-{current_group}",
            order=order,
            raw_command=line,
        )
        existing = seen.get(key)
        if existing is not None:
            existing.aliases.append(filename)
            duplicate_notes.append(
                f"- removed duplicate {resource.kind}: {resource.source} (alias {filename})"
            )
            continue
        seen[key] = resource
        resources.append(resource)
        order += 1

    return resources, duplicate_notes


def render_clean_markdown(resources: list[Resource], duplicate_notes: list[str]) -> str:
    by_kind = {kind: [r for r in resources if r.kind == kind] for kind in ("paper", "repo", "package")}
    lines: list[str] = [
        "# Quant Resource Bundle - Clean Inventory",
        "",
        f"Generated from `{BUNDLE_PATH.relative_to(ROOT)}`.",
        "",
        "## Summary",
        f"- Unique resources: {len(resources)}",
        f"- Papers: {len(by_kind['paper'])}",
        f"- Repos: {len(by_kind['repo'])}",
        f"- Packages: {len(by_kind['package'])}",
        f"- Download root: `{ARTIFACT_ROOT.relative_to(ROOT)}`",
        f"- Canonical paper root: `docs/references/papers`",
        "- Torch-first policy: TensorFlow is excluded from the default artifact set. Set `ALLOW_TENSORFLOW=1` only if you explicitly want to override that choice.",
        "",
        "## Deduping Rules",
        "- Exact duplicate repository URLs were removed.",
        "- Exact duplicate paper source URLs were removed.",
        "- Package install lines were preserved as distinct resources unless the line itself repeated.",
        "- When a duplicate appeared, the first occurrence was kept and later aliases were recorded in the manifest.",
        "",
        "## Priority Model",
        "- Priority 1: direct execution, portfolio, and trading/HFT resources.",
        "- Priority 2: supporting time-series, pricing, and deep learning resources.",
        "- Priority 3: bonus lists and secondary archives.",
        "",
    ]

    if duplicate_notes:
        lines.append("## Removed Duplicates")
        lines.extend(duplicate_notes)
        lines.append("")

    for kind in ("paper", "repo", "package"):
        kind_items = sorted(
            by_kind[kind],
            key=lambda item: (item.priority, item.section, item.order),
        )
        lines.append(f"## {kind.title()}s")
        lines.append("| Priority | Label | Source | Destination | Section |")
        lines.append("| --- | --- | --- | --- | --- |")
        for item in kind_items[:12]:
            lines.append(
                f"| {item.priority} | {item.label} | {item.source} | `{item.destination}` | {item.section} |"
            )
        if len(kind_items) > 12:
            lines.append(f"| ... | ... | ... | ... | {len(kind_items) - 12} more items |")
        lines.append("")

    lines.extend(
        [
            "## Full Manifest",
            f"- Machine-readable manifest: `{MANIFEST_PATH.relative_to(ROOT)}`",
            f"- Executable downloader: `{DOWNLOAD_SCRIPT_PATH.relative_to(ROOT)}`",
            "",
            "Run the downloader with for example:",
            "",
            "```bash",
            "PRIORITY_MAX=1 bash scripts/download_quant_resource_bundle.sh",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def render_download_script(resources: list[Resource], duplicate_notes: list[str]) -> str:
    ordered = sorted(resources, key=lambda item: (item.priority, KIND_ORDER[item.kind], item.section, item.order))
    lines: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
        'ARTIFACT_ROOT="${ARTIFACT_ROOT:-$ROOT_DIR/artifacts/quant_resource_bundle}"',
        'CANONICAL_PAPER_ROOT="${CANONICAL_PAPER_ROOT:-$ROOT_DIR/docs/references/papers}"',
        'PRIORITY_MAX="${PRIORITY_MAX:-3}"',
        'KIND_FILTER="${KIND_FILTER:-all}"',
        'MODE="${MODE:-download}"',
        'FORCE="${FORCE:-0}"',
        'PYTHON_BIN="${PYTHON_BIN:-python3}"',
        "",
        "ensure_paper_root() {",
        "  mkdir -p \"$CANONICAL_PAPER_ROOT\"",
        "  if [[ -d \"$ARTIFACT_ROOT/papers\" && ! -L \"$ARTIFACT_ROOT/papers\" ]]; then",
        "    shopt -s nullglob",
        "    local existing_papers=(\"$ARTIFACT_ROOT/papers\"/*.pdf)",
        "    shopt -u nullglob",
        "    if ((${#existing_papers[@]} > 0)); then",
        "      mv \"${existing_papers[@]}\" \"$CANONICAL_PAPER_ROOT\"/",
        "    fi",
        "    rmdir \"$ARTIFACT_ROOT/papers\" 2>/dev/null || true",
        "  elif [[ -e \"$ARTIFACT_ROOT/papers\" && ! -L \"$ARTIFACT_ROOT/papers\" ]]; then",
        "    rm -rf \"$ARTIFACT_ROOT/papers\"",
        "  fi",
        "  if [[ ! -e \"$ARTIFACT_ROOT/papers\" ]]; then",
        "    ln -s \"$CANONICAL_PAPER_ROOT\" \"$ARTIFACT_ROOT/papers\"",
        "  fi",
        "}",
        "",
        "ensure_paper_root",
        "mkdir -p \"$ARTIFACT_ROOT/repos\" \"$ARTIFACT_ROOT/pypi\" \"$ARTIFACT_ROOT/logs\"",
        "",
        "success_count=0",
        "skip_count=0",
        "fail_count=0",
        "paper_count=0",
        "repo_count=0",
        "package_count=0",
        "",
        "log() {",
        "  printf '[%s] %s\\n' \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\" \"$*\"",
        "}",
        "",
        "should_run() {",
        "  local priority=\"$1\"",
        "  local kind=\"$2\"",
        "  [[ \"$priority\" -le \"$PRIORITY_MAX\" ]] || return 1",
        "  [[ \"$KIND_FILTER\" == \"all\" || \"$KIND_FILTER\" == \"$kind\" ]] || return 1",
        "  return 0",
        "}",
        "",
        "run_paper() {",
        "  local priority=\"$1\"",
        "  local label=\"$2\"",
        "  local dest_rel=\"$3\"",
        "  local source=\"$4\"",
        "  local dest=\"$ARTIFACT_ROOT/$dest_rel\"",
        "  if ! should_run \"$priority\" paper; then",
        "    ((skip_count+=1))",
        "    return 0",
        "  fi",
        "  if [[ -s \"$dest\" && \"$FORCE\" != \"1\" ]]; then",
        "    log \"paper skip: $label\"",
        "    ((skip_count+=1))",
        "    ((paper_count+=1))",
        "    return 0",
        "  fi",
        "  log \"paper fetch: $label\"",
        "  if wget -q --show-progress --tries=3 --timeout=60 --waitretry=2 -U 'Mozilla/5.0' -O \"$dest\" \"$source\"; then",
        "    ((success_count+=1))",
        "  else",
        "    ((fail_count+=1))",
        "  fi",
        "  ((paper_count+=1))",
        "}",
        "",
        "run_repo() {",
        "  local priority=\"$1\"",
        "  local label=\"$2\"",
        "  local dest_rel=\"$3\"",
        "  local source=\"$4\"",
        "  local dest=\"$ARTIFACT_ROOT/$dest_rel\"",
        "  if ! should_run \"$priority\" repo; then",
        "    ((skip_count+=1))",
        "    return 0",
        "  fi",
        "  if [[ -e \"$dest\" && \"$FORCE\" != \"1\" ]]; then",
        "    log \"repo skip: $label\"",
        "    ((skip_count+=1))",
        "    ((repo_count+=1))",
        "    return 0",
        "  fi",
        "  if [[ -e \"$dest\" && \"$FORCE\" == \"1\" ]]; then",
        "    rm -rf \"$dest\"",
        "  fi",
        "  mkdir -p \"$(dirname \"$dest\")\"",
        "  log \"repo fetch: $label\"",
        "  if GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --single-branch \"$source\" \"$dest\"; then",
        "    ((success_count+=1))",
        "  else",
        "    ((fail_count+=1))",
        "  fi",
        "  ((repo_count+=1))",
        "}",
        "",
        "run_package() {",
        "  local priority=\"$1\"",
        "  local label=\"$2\"",
        "  local dest_rel=\"$3\"",
        "  shift 3",
        "  local dest=\"$ARTIFACT_ROOT/$dest_rel\"",
        "  if ! should_run \"$priority\" package; then",
        "    ((skip_count+=1))",
        "    return 0",
        "  fi",
        "  if [[ \"$label\" == \"tensorflow\" && \"${ALLOW_TENSORFLOW:-0}\" != \"1\" ]]; then",
        "    log \"package skip: $label (torch-first policy)\"",
        "    ((skip_count+=1))",
        "    ((package_count+=1))",
        "    return 0",
        "  fi",
        "  if [[ -e \"$dest\" && \"$FORCE\" != \"1\" ]]; then",
        "    log \"package skip: $label\"",
        "    ((skip_count+=1))",
        "    ((package_count+=1))",
        "    return 0",
        "  fi",
        "  mkdir -p \"$dest\"",
        "  log \"package fetch: $label\"",
        "  if [[ \"$MODE\" == \"install\" ]]; then",
        "    if \"$PYTHON_BIN\" -m pip install --disable-pip-version-check \"$@\"; then",
        "      ((success_count+=1))",
        "    else",
        "      ((fail_count+=1))",
        "    fi",
        "  else",
        "    if \"$PYTHON_BIN\" -m pip download --no-deps --disable-pip-version-check --dest \"$dest\" \"$@\"; then",
        "      ((success_count+=1))",
        "    else",
        "      ((fail_count+=1))",
        "    fi",
        "  fi",
        "  ((package_count+=1))",
        "}",
        "",
    ]

    if duplicate_notes:
        lines.append("# Duplicate notes preserved from the generator.")
        for note in duplicate_notes:
            lines.append(f"# {note}")
        lines.append("")

    current_priority: int | None = None
    current_kind: str | None = None
    for item in ordered:
        if item.priority != current_priority:
            current_priority = item.priority
            lines.append(f"# Priority {current_priority}")
        if item.kind != current_kind:
            current_kind = item.kind
            lines.append(f"# {current_kind.title()}s")
        if item.kind == "paper":
            lines.append(
                "run_paper "
                + " ".join(
                    [
                        str(item.priority),
                        shell_quote(item.label),
                        shell_quote(item.destination),
                        shell_quote(item.source),
                    ]
                )
            )
        elif item.kind == "repo":
            lines.append(
                "run_repo "
                + " ".join(
                    [
                        str(item.priority),
                        shell_quote(item.label),
                        shell_quote(item.destination),
                        shell_quote(item.source),
                    ]
                )
            )
        else:
            specs = shlex.split(item.source)
            quoted_specs = " ".join(shell_quote(spec) for spec in specs)
            lines.append(
                "run_package "
                + " ".join(
                    [
                        str(item.priority),
                        shell_quote(item.label),
                        shell_quote(item.destination),
                        quoted_specs,
                    ]
                )
            )

    lines.extend(
        [
            "",
            "summary_path=\"$ARTIFACT_ROOT/run_summary.json\"",
            "cat > \"$summary_path\" <<EOF",
            "{",
            '  "downloaded_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",',
            '  "priority_max": $PRIORITY_MAX,',
            '  "kind_filter": "$KIND_FILTER",',
            '  "mode": "$MODE",',
            '  "success_count": $success_count,',
            '  "skip_count": $skip_count,',
            '  "fail_count": $fail_count,',
            '  "paper_count": $paper_count,',
            '  "repo_count": $repo_count,',
            '  "package_count": $package_count',
            "}",
            "EOF",
            "log \"summary written to $summary_path\"",
            "log \"success=$success_count skip=$skip_count fail=$fail_count\"",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    text = BUNDLE_PATH.read_text(encoding="utf-8")
    resources, duplicate_notes = first_pass_resources(text)
    resources.sort(key=lambda item: (item.priority, KIND_ORDER[item.kind], item.section, item.order))

    manifest = {
        "source": str(BUNDLE_PATH.relative_to(ROOT)),
        "artifact_root": str(ARTIFACT_ROOT.relative_to(ROOT)),
        "duplicate_notes": duplicate_notes,
        "counts": {
            "total": len(resources),
            "paper": sum(1 for item in resources if item.kind == "paper"),
            "repo": sum(1 for item in resources if item.kind == "repo"),
            "package": sum(1 for item in resources if item.kind == "package"),
        },
        "items": [item.as_manifest_item() for item in resources],
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    CLEAN_MD_PATH.write_text(render_clean_markdown(resources, duplicate_notes), encoding="utf-8")
    DOWNLOAD_SCRIPT_PATH.write_text(render_download_script(resources, duplicate_notes), encoding="utf-8")
    DOWNLOAD_SCRIPT_PATH.chmod(0o755)

    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"Wrote {CLEAN_MD_PATH.relative_to(ROOT)}")
    print(f"Wrote {DOWNLOAD_SCRIPT_PATH.relative_to(ROOT)}")
    print(f"Unique resources: {len(resources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
