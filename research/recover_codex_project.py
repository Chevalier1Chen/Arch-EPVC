"""Rebuild the deleted Arch-EPVC text source tree from Codex patch history.

The recovery is deterministic and read-only with respect to the session log. It
replays successful apply_patch events in chronological order and writes the
latest surviving version of every project file into the current workspace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath


SESSION_LOG = Path(
    "C:/Users/DELL/.codex/sessions/2026/07/31/"
    "rollout-2026-07-31T11-58-57-019fb653-43ca-7a30-9493-f0d87bd84060.jsonl"
)
PROJECT_MARKER = "school-building-design-platform"
OUTPUT_ROOT = Path(__file__).resolve().parent / PROJECT_MARKER
HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$"
)


def apply_unified_diff(source: str, diff: str, path: str) -> str:
    """Apply the headerless unified diff emitted by Codex apply_patch."""
    old_lines = source.replace("\r\n", "\n").splitlines()
    diff_lines = diff.replace("\r\n", "\n").splitlines()
    result: list[str] = []
    old_cursor = 0
    index = 0

    while index < len(diff_lines):
        match = HUNK_RE.match(diff_lines[index])
        if not match:
            index += 1
            continue

        old_start = int(match.group(1))
        hunk_cursor = max(old_start - 1, 0)
        if hunk_cursor < old_cursor:
            raise ValueError(f"Overlapping hunk in {path}")
        result.extend(old_lines[old_cursor:hunk_cursor])
        old_cursor = hunk_cursor
        index += 1

        while index < len(diff_lines) and not diff_lines[index].startswith("@@ "):
            line = diff_lines[index]
            index += 1
            if line == r"\ No newline at end of file":
                continue
            if not line:
                raise ValueError(f"Malformed empty diff line in {path}")
            marker, text = line[0], line[1:]
            if marker == " ":
                if old_cursor >= len(old_lines) or old_lines[old_cursor] != text:
                    raise ValueError(
                        f"Context mismatch in {path} at source line {old_cursor + 1}"
                    )
                result.append(text)
                old_cursor += 1
            elif marker == "-":
                if old_cursor >= len(old_lines) or old_lines[old_cursor] != text:
                    raise ValueError(
                        f"Removal mismatch in {path} at source line {old_cursor + 1}"
                    )
                old_cursor += 1
            elif marker == "+":
                result.append(text)
            else:
                raise ValueError(f"Unknown diff marker {marker!r} in {path}")

    result.extend(old_lines[old_cursor:])
    # Codex-created source files in this project consistently use a final LF.
    return "\n".join(result) + ("\n" if result else "")


def relative_project_path(raw_path: str) -> Path | None:
    parts = PureWindowsPath(raw_path).parts
    try:
        marker_index = next(
            index for index, part in enumerate(parts) if part == PROJECT_MARKER
        )
    except StopIteration:
        return None
    tail = parts[marker_index + 1 :]
    return Path(*tail) if tail else Path()


def recover() -> tuple[int, list[str]]:
    states: dict[Path, str] = {}
    deleted: set[Path] = set()
    unresolved: list[str] = []
    patch_events = 0

    with SESSION_LOG.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            if '"type":"patch_apply_end"' not in raw_line:
                continue
            event = json.loads(raw_line)
            payload = event.get("payload", {})
            timestamp = event.get("timestamp", "unknown-time")
            if not payload.get("success") or not payload.get("changes"):
                continue
            patch_events += 1

            for raw_path, change in payload["changes"].items():
                rel_path = relative_project_path(raw_path)
                if rel_path is None or not rel_path.parts:
                    continue
                kind = change.get("type")
                label = str(rel_path).replace("\\", "/")

                if kind == "add":
                    states[rel_path] = change.get("content", "")
                    deleted.discard(rel_path)
                elif kind == "delete":
                    states.pop(rel_path, None)
                    deleted.add(rel_path)
                elif kind == "update":
                    if rel_path not in states:
                        unresolved.append(f"{timestamp} missing base: {label}")
                        continue
                    try:
                        states[rel_path] = apply_unified_diff(
                            states[rel_path], change.get("unified_diff", ""), label
                        )
                    except ValueError as error:
                        unresolved.append(
                            f"{timestamp} {error}; diff={change.get('unified_diff', '')!r}"
                        )
                else:
                    unresolved.append(
                        f"{timestamp} unsupported change {kind!r}: {label}"
                    )

                move_path = change.get("move_path")
                if move_path and rel_path in states:
                    moved_rel = relative_project_path(move_path)
                    if moved_rel is not None:
                        states[moved_rel] = states.pop(rel_path)
                        deleted.add(rel_path)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for rel_path, content in states.items():
        destination = OUTPUT_ROOT / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")

    report = OUTPUT_ROOT / "RECOVERY_REPORT.txt"
    report.write_text(
        "\n".join(
            [
                f"Patch events scanned: {patch_events}",
                f"Recovered text files: {len(states)}",
                f"Deleted files omitted: {len(deleted)}",
                f"Unresolved operations: {len(unresolved)}",
                "",
                *unresolved,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return len(states), unresolved


if __name__ == "__main__":
    recovered_count, problems = recover()
    print(f"Recovered {recovered_count} text files into {OUTPUT_ROOT}")
    print(f"Unresolved operations: {len(problems)}")
