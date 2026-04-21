import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


CAPTION_FIELDS = ["human_caption", "gemini_caption"]


@dataclass
class SampleRecord:
    key: str
    image_path: Optional[str]
    caption: Optional[str]
    caption_source: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict)


def _resolve_caption(
    entry: Dict[str, Any],
    caption_field: Optional[str],
    skip_token: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    fields_to_try = [caption_field] if caption_field else CAPTION_FIELDS

    for field_name in fields_to_try:
        raw_val = (entry.get(field_name) or "").strip()
        if not raw_val:
            continue
        if skip_token and raw_val.lower() == skip_token.lower():
            continue
        return raw_val, field_name

    return None, None


def load_records(
    input_json: str,
    caption_field: Optional[str] = None,
    skip_token: Optional[str] = "reject",
    require_image_exists: bool = False,
) -> List[SampleRecord]:
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with open(input_json, "r") as f:
        data: Dict[str, Any] = json.load(f)

    records: List[SampleRecord] = []

    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue

        caption, caption_source = _resolve_caption(entry, caption_field, skip_token)
        image_path: Optional[str] = entry.get("frame_path")

        if require_image_exists and (not image_path or not os.path.exists(image_path)):
            continue

        records.append(SampleRecord(
            key=key,
            image_path=image_path,
            caption=caption,
            caption_source=caption_source,
            raw=entry,
        ))

    return records
