"""
Helpers for overlaying empirical sim projections onto a DraftKings slate.

The optimizer keeps salary, ownership, and contest metadata from the DK slate
while using empirical outputs from the game-sim layer for mean and percentile
columns. This module normalizes those fields so the existing loaders can read
them without depending on one exact schema.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


_PERCENTILE_ALIASES = {
    "dk_p10": "dk_10_percentile",
    "dk_p25": "dk_25_percentile",
    "dk_p50": "dk_50_percentile",
    "dk_p75": "dk_75_percentile",
    "dk_p85": "dk_85_percentile",
    "dk_p90": "dk_90_percentile",
    "dk_p95": "dk_95_percentile",
    "dk_p99": "dk_99_percentile",
}


def _normalize_text(value) -> str:
    return " ".join(str(value).strip().lower().split())


def _row_candidate_keys(row: Mapping[str, object]) -> List[str]:
    keys: List[str] = []

    for field in ("player_id", "DFS ID", "DFS ID ", "ID", "Name", "name", "Name + ID"):
        raw = row.get(field)
        if raw in (None, ""):
            continue
        text = str(raw).strip()
        if not text:
            continue
        keys.append(_normalize_text(text))

        match = re.search(r"\(([^)]+)\)", text)
        if match:
            keys.append(_normalize_text(match.group(1)))

        if field == "Name + ID":
            parts = text.rsplit("(", 1)
            if len(parts) == 2:
                keys.append(_normalize_text(parts[0]))
                keys.append(_normalize_text(parts[1].rstrip(")")))

    return list(dict.fromkeys(keys))


def load_projection_rows(path: str) -> List[Dict[str, object]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(dict(payload))
        return rows

    if suffix == ".json":
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            if "projections" in payload and isinstance(payload["projections"], list):
                rows = payload["projections"]
            else:
                rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        return [dict(row) for row in rows if isinstance(row, dict)]

    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _normalize_overlay_row(row: Mapping[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = dict(row)

    # Keep the empirical sim fields available under the legacy names that the
    # current optimizer already knows how to read.
    if normalized.get("dk_mean") not in (None, ""):
        normalized.setdefault("dk_points", normalized["dk_mean"])
    if normalized.get("dk_median") not in (None, ""):
        normalized.setdefault("dk_median", normalized["dk_median"])

    for source, target in _PERCENTILE_ALIASES.items():
        if normalized.get(source) not in (None, ""):
            normalized.setdefault(target, normalized[source])
            normalized.setdefault(source, normalized[source])

    # Make sure the base loaders can read team/opponent fields from either the
    # DK slate or the sim overlay.
    if normalized.get("team") not in (None, ""):
        normalized.setdefault("Team", normalized["team"])
        normalized.setdefault("TeamAbbrev", normalized["team"])
    if normalized.get("opp") not in (None, ""):
        normalized.setdefault("Opp", normalized["opp"])

    if normalized.get("player_id") not in (None, ""):
        normalized.setdefault("DFS ID", normalized["player_id"])
        normalized.setdefault("ID", normalized["player_id"])
    if normalized.get("name") not in (None, ""):
        normalized.setdefault("Name", normalized["name"])

    return normalized


def merge_projection_rows(
    base_rows: Sequence[Mapping[str, object]],
    overlay_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """
    Merge empirical sim projections into an existing DK slate row set.

    The base row wins for salary, ownership, position, and contest metadata.
    The overlay row wins for empirical distribution fields.
    """
    overlay_index: Dict[str, Dict[str, object]] = {}
    normalized_overlay = [_normalize_overlay_row(row) for row in overlay_rows]
    for row in normalized_overlay:
        for key in _row_candidate_keys(row):
            overlay_index.setdefault(key, row)

    merged_rows: List[Dict[str, object]] = []
    for base in base_rows:
        merged = dict(base)
        for key in _row_candidate_keys(base):
            overlay = overlay_index.get(key)
            if not overlay:
                continue

            for field, value in overlay.items():
                if field in {
                    "dk_mean",
                    "dk_median",
                    "dk_points",
                    "dk_std",
                    "dk_p10",
                    "dk_p25",
                    "dk_p50",
                    "dk_p75",
                    "dk_p85",
                    "dk_p90",
                    "dk_p95",
                    "dk_p99",
                    "dk_10_percentile",
                    "dk_25_percentile",
                    "dk_50_percentile",
                    "dk_75_percentile",
                    "dk_85_percentile",
                    "dk_90_percentile",
                    "dk_95_percentile",
                    "dk_99_percentile",
                    "team",
                    "opp",
                    "Team",
                    "TeamAbbrev",
                    "Opp",
                    "player_id",
                    "name",
                    "DFS ID",
                    "ID",
                }:
                    if value not in (None, ""):
                        merged[field] = value
            break

        merged_rows.append(merged)

    return merged_rows
