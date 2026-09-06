from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from fastapi import HTTPException


def normalize_candidate_id(value: object) -> str:
    return Path(str(value)).stem


def parse_candidate_names(raw: str | None) -> list[str] | None:
    """Parse an optional multipart JSON list of unique outfit IDs."""
    if raw is None or not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="candidate_names must be valid JSON") from exc
    if not isinstance(payload, list) or not payload or not all(isinstance(item, str) and item.strip() for item in payload):
        raise HTTPException(status_code=422, detail="candidate_names must be a non-empty JSON array of strings")
    names = [normalize_candidate_id(item) for item in payload]
    if len(names) != len(set(names)):
        raise HTTPException(status_code=422, detail="candidate_names contains duplicate outfit IDs")
    return names


def resolve_candidate_filenames(candidate_ids: Sequence[str], directory: Path) -> list[str]:
    """Resolve IDs to exact filenames and reject unknown or ambiguous IDs."""
    by_stem: dict[str, list[str]] = {}
    for path in directory.iterdir():
        if path.is_file():
            by_stem.setdefault(path.stem, []).append(path.name)
    missing = [item_id for item_id in candidate_ids if item_id not in by_stem]
    ambiguous = [item_id for item_id in candidate_ids if len(by_stem.get(item_id, [])) > 1]
    if missing or ambiguous:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid candidate pool: missing={missing}, ambiguous={ambiguous}",
        )
    return [by_stem[item_id][0] for item_id in candidate_ids]
