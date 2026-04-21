import os
import json
import shutil
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

from load_json_data import load_records


DEFAULT_MODEL_ID = "google/siglip2-so400m-patch14-384"
DEFAULT_OUTPUT_DIR = "./siglip_scores"
DEFAULT_BATCH_SIZE = 64


def load_siglip2(model_id: str) -> Tuple[Any, Any, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor, device


def _load_image(image_path: str) -> Image.Image:
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return Image.new("RGB", (384, 384), (0, 0, 0))


def compute_siglip2_scores_batch(
    model,
    processor,
    device: str,
    image_paths: List[str],
    captions: List[str],
) -> List[float]:
    images = [_load_image(p) for p in image_paths]

    inputs = processor(
        text=captions,
        images=images,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    ).to(device)

    with torch.no_grad(), torch.autocast(device_type=device, enabled=(device == "cuda")):
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits_per_image.diagonal()).cpu().tolist()

    return scores


def duplicate_json(src: str, dst: str):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"Duplicated input JSON to: {dst}")
    else:
        print(f"Output JSON already exists, skipping copy: {dst}")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def load_done_keys(events_jsonl: str) -> set:
    done = set()
    if not os.path.exists(events_jsonl):
        return done
    with open(events_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("key")
                if k:
                    done.add(k)
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser(description="Score image-caption pairs with SigLIP 2")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_json", type=str, default="data_with_siglip_scores.json")
    parser.add_argument("--events_jsonl", type=str, default="siglip_events.jsonl")
    parser.add_argument("--caption_field", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--snapshot_every", type=int, default=100)
    parser.add_argument("--skip_token", type=str, default="reject")
    args = parser.parse_args()

    output_json_path = os.path.join(args.output_dir, args.output_json)
    events_jsonl_path = os.path.join(args.output_dir, args.events_jsonl)
    skip_token = args.skip_token or None

    duplicate_json(args.input_json, output_json_path)
    output_data = load_json(output_json_path)

    records = load_records(
        args.input_json,
        caption_field=args.caption_field,
        skip_token=skip_token,
        require_image_exists=False,
    )
    print(f"Loaded {len(records)} records from: {args.input_json}")

    done_keys = load_done_keys(events_jsonl_path)
    if done_keys:
        print(f"Resuming - {len(done_keys)} keys already processed.")

    print(f"Loading SigLIP 2 model: {args.model_id}")
    model, processor, device = load_siglip2(args.model_id)

    null_records: List[Tuple[Any, str]] = []
    score_records: List[Any] = []

    for record in records:
        if record.key in done_keys:
            continue
        if record.caption is None:
            null_records.append((record, "no_valid_caption"))
        elif not record.image_path or not os.path.exists(record.image_path):
            null_records.append((record, "missing_image"))
        else:
            score_records.append(record)

    print(f"  {len(null_records)} records skipped (no caption / missing image)")
    print(f"  {len(score_records)} records to score in batches of {args.batch_size}")

    os.makedirs(args.output_dir, exist_ok=True)
    processed = 0
    batches_done = 0

    with open(events_jsonl_path, "a") as ef:
        for record, reason in null_records:
            output_data[record.key]["siglip_score"] = None
            ef.write(json.dumps({"key": record.key, "siglip_score": None, "reason": reason}) + "\n")
            processed += 1
        ef.flush()

        n_batches = (len(score_records) + args.batch_size - 1) // args.batch_size
        for batch_idx in tqdm(range(n_batches), desc="Scoring SigLIP 2 (batches)"):
            start = batch_idx * args.batch_size
            batch = score_records[start: start + args.batch_size]

            image_paths = [r.image_path for r in batch]
            captions = [r.caption for r in batch]

            try:
                scores = compute_siglip2_scores_batch(model, processor, device, image_paths, captions)
                for record, score in zip(batch, scores):
                    output_data[record.key]["siglip_score"] = score
                    ef.write(json.dumps({"key": record.key, "siglip_score": score}) + "\n")
            except Exception as e:
                err_str = str(e)
                for record in batch:
                    output_data[record.key]["siglip_score"] = None
                    ef.write(json.dumps({"key": record.key, "siglip_score": None, "error": err_str}) + "\n")

            ef.flush()
            processed += len(batch)
            batches_done += 1

            if args.snapshot_every > 0 and batches_done % args.snapshot_every == 0:
                write_json(output_json_path, output_data)

    write_json(output_json_path, output_data)

    print("\nDone.")
    print(f"Scored JSON: {os.path.abspath(output_json_path)}")
    print(f"Event log: {os.path.abspath(events_jsonl_path)}")
    print(f"Records processed: {processed}")


if __name__ == "__main__":
    main()
