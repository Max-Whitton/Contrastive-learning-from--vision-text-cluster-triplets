import json
import os
import argparse
from tqdm import tqdm
from clip_filter import load_model, compute_similarity

MODEL_NAME = "ViT-H-14-quickgelu"
PRETRAINED = "dfn5b"

CAPTION_KEYS = (
    "human_caption",
    "audio_caption",
    "gemini_caption",
    "caption",
    "text",
    "touch_caption",
    "description",
    "label",
)
IMAGE_KEYS = (
    "frame_path",
    "image_path",
    "path",
    "file_path",
    "image",
    "img_path",
    "jpg_path",
    "abs_path",
    "video_path",
    "clip_path",
)


def _caption_text(entry):
    if not isinstance(entry, dict):
        return ""
    for k in CAPTION_KEYS:
        v = entry.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    hl = entry.get("human_labels")
    if isinstance(hl, str) and str(hl).strip():
        return str(hl).strip()
    if isinstance(hl, dict):
        for k in CAPTION_KEYS:
            v = hl.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
    return ""


def _image_path(entry, base_dir=None):
    if not isinstance(entry, dict):
        raise KeyError("entry is not a JSON object")
    raw = None
    for k in IMAGE_KEYS:
        v = entry.get(k)
        if v is not None and str(v).strip():
            raw = str(v).strip()
            break
    if raw is None:
        raise KeyError(
            f"no image path (add one of {IMAGE_KEYS}); keys present: {list(entry.keys())[:20]}"
        )
    if os.path.isabs(raw):
        return raw
    if base_dir is None:
        return raw
    stem = os.path.basename(raw)
    name_no_ext = os.path.splitext(stem)[0]
    jpg_candidate = os.path.join(base_dir, "center_frames", name_no_ext + ".jpg")
    if os.path.exists(jpg_candidate):
        return jpg_candidate
    direct = os.path.join(base_dir, raw)
    if os.path.exists(direct):
        return direct
    parts = stem.split("_")
    if len(parts) >= 4:
        subdir = "_".join(parts[:4])
        candidate = os.path.join(base_dir, subdir, raw)
        if os.path.exists(candidate):
            return candidate
    return direct


def _iter_entries(data):
    if isinstance(data, dict):
        for k, entry in data.items():
            yield str(k), entry
    elif isinstance(data, list):
        for i, entry in enumerate(data):
            yield str(i), entry
    else:
        raise TypeError(f"JSON root must be dict or list, got {type(data).__name__}")


def main():
    parser = argparse.ArgumentParser(description="Score labels with CLIP")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=None)
    args = parser.parse_args()

    if args.input == args.output:
        print("ERROR: --output must be different from --input!")
        return

    with open(args.input) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {args.input}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model, preprocess, tokenizer = load_model(MODEL_NAME, PRETRAINED, cache_dir=args.cache_dir)

    scored = 0
    rejected = 0
    errors = 0

    for label, entry in tqdm(_iter_entries(data), total=len(data), desc="Scoring"):
        if not isinstance(entry, dict):
            errors += 1
            if errors <= 3:
                print(f"  Error on {label}: entry is not an object, skipping")
            continue

        caption = _caption_text(entry)

        if caption.lower() == "reject":
            entry["clip_score"] = -1
            rejected += 1
            continue

        if not caption:
            entry["clip_score"] = -2
            errors += 1
            if errors <= 3:
                print(
                    f"  Error on {label}: no caption field (tried {CAPTION_KEYS}); "
                    f"keys={list(entry.keys())}"
                )
            continue

        try:
            img_path = _image_path(entry, base_dir=args.base_dir)
            scores = compute_similarity(
                model, preprocess, tokenizer,
                img_path, caption
            )
            entry["clip_score"] = round(scores[0], 4)
            scored += 1
        except Exception as e:
            entry["clip_score"] = -2
            errors += 1
            if errors <= 3:
                print(f"  Error on {label}: {e}")

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone!")
    print(f"  Scored: {scored}")
    print(f"  Rejected (set to -1): {rejected}")
    print(f"  Errors (set to -2): {errors}")
    print(f"  Saved to: {args.output}")

    all_entries = data.values() if isinstance(data, dict) else data
    valid_scores = [
        e["clip_score"] for e in all_entries
        if isinstance(e, dict) and e.get("clip_score", 0) > 0
    ]
    if valid_scores:
        valid_scores.sort()
        print(f"\n  Score stats (non-rejected):")
        print(f"    Count: {len(valid_scores)}")
        print(f"    Min:   {min(valid_scores):.4f}")
        print(f"    Max:   {max(valid_scores):.4f}")
        print(f"    Mean:  {sum(valid_scores)/len(valid_scores):.4f}")
        print(f"    Median:{valid_scores[len(valid_scores)//2]:.4f}")


if __name__ == "__main__":
    main()
