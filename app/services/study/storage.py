"""
Study storage service.
Canonical location: app/services/study/storage.py
(Backward-compat shim remains at app/services/study_storage.py)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_response(payload: Dict[str, Any], *, directory_path: str) -> Dict[str, Any]:
    """Write a participant payload to its own JSON file named by participantId."""
    os.makedirs(directory_path, exist_ok=True)
    participant_id = payload.get("participantId") or payload.get("participant_id")
    if not participant_id:
        raise ValueError("payload must include participantId")
    record = {"receivedAt": _utc_now_iso(), "payload": payload}
    file_path = os.path.join(directory_path, f"{participant_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return {"file_path": file_path, "participantId": participant_id, "receivedAt": record["receivedAt"]}


def read_all_payloads(*, directory_path: str) -> List[Dict[str, Any]]:
    """Read all stored participant payloads from per-participant JSON files."""
    if not os.path.isdir(directory_path):
        return []
    payloads: List[Dict[str, Any]] = []
    for filename in sorted(os.listdir(directory_path)):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(directory_path, filename)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                record = json.load(f)
            except (json.JSONDecodeError, Exception):
                continue
        payload = record.get("payload") if isinstance(record, dict) else None
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def try_find_by_participant_id(*, directory_path: str, participant_id: str) -> Optional[Dict[str, Any]]:
    """Return the stored payload for a participantId, or None if not found."""
    file_path = os.path.join(directory_path, f"{participant_id}.json")
    if not os.path.isfile(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            record = json.load(f)
        except (json.JSONDecodeError, Exception):
            return None
    payload = record.get("payload") if isinstance(record, dict) else None
    return payload if isinstance(payload, dict) else None
