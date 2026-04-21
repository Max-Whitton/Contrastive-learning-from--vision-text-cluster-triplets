import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval


DEFAULT_MODEL_ID = "Salesforce/blip-itm-base-coco"
DEFAULT_OUTPUT_DIR = "./blip_itm_scores"
DEFAULT_NUM_FRAMES = 4


def load_blip_itm(model_id: str) -> Tuple[Any, Any, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForImageTextRetrieval.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor, device


def extract_frames(video_path: str, num_frames: int = 4) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return [Image.new("RGB", (384, 384), (0, 0, 0))]

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return [Image.new("RGB", (384, 384), (0, 0, 0))]

    start = max(0, int(total_frames * 0.05))
    end = min(total_frames - 1, int(total_frames * 0.95))
    if end <= start:
        indices = [total_frames // 2]
    else:
        indices = np.linspace(start, end, num=min(num_frames, total_frames), dtype=int).tolist()

    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()

    if not frames:
        return [Image.new("RGB", (384, 384), (0, 0, 0))]
    return frames


def compute_blip_itm_scores_batch(
    model,
    processor,
    device: str,
    images: List[Image.Image],
    captions: List[str],
) -> List[float]:
    inputs = processor(
        images=images,
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, use_itm_head=True)
        scores = F.softmax(outputs.itm_score, dim=1)[:, 1].cpu().tolist()

    return scores


def score_video(
    model,
    processor,
    device: str,
    video_path: str,
    caption: str,
    num_frames: int,
    frame_batch_size: int = 16,
) -> Dict[str, Any]:
    frames = extract_frames(video_path, num_frames=num_frames)

    all_scores: List[float] = []
    for i in range(0, len(frames), frame_batch_size):
        batch_frames = frames[i : i + frame_batch_size]
        batch_captions = [caption] * len(batch_frames)
        scores = compute_blip_itm_scores_batch(model, processor, device, batch_frames, batch_captions)
        all_scores.extend(scores)

    return {
        "mean": float(np.mean(all_scores)),
        "max": float(np.max(all_scores)),
        "min": float(np.min(all_scores)),
        "per_frame": all_scores,
        "num_frames_scored": len(all_scores),
    }


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: Any):
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
                if k is not None:
                    done.add(k)
            except Exception:
                pass
    return done


def main():
    parser = argparse.ArgumentParser(description="Score video-caption pairs with BLIP ITM")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_json", type=str, default="data_with_blip_scores.json")
    parser.add_argument("--events_jsonl", type=str, default="blip_events.jsonl")
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--snapshot_every", type=int, default=100)
    args = parser.parse_args()

    output_json_path = os.path.join(args.output_dir, args.output_json)
    events_jsonl_path = os.path.join(args.output_dir, args.events_jsonl)
    video_dir = args.video_dir or os.path.dirname(os.path.abspath(args.input_json))

    input_data: List[Dict[str, Any]] = load_json(args.input_json)
    print(f"Loaded {len(input_data)} records from: {args.input_json}")

    done_keys = load_done_keys(events_jsonl_path)
    if done_keys:
        print(f"Resuming - {len(done_keys)} records already processed.")

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(output_json_path):
        output_data: List[Dict[str, Any]] = load_json(output_json_path)
        print(f"Loaded existing output JSON with {len(output_data)} records.")
    else:
        import copy
        output_data = copy.deepcopy(input_data)

    output_lookup: Dict[str, int] = {}
    for idx, rec in enumerate(output_data):
        output_lookup[rec["video_path"]] = idx

    print(f"Loading BLIP ITM model: {args.model_id}")
    model, processor, device = load_blip_itm(args.model_id)

    null_records: List[Tuple[int, Dict, str]] = []
    score_records: List[Tuple[int, Dict]] = []

    for idx, record in enumerate(input_data):
        key = record["video_path"]
        if key in done_keys:
            continue

        caption = record.get("touch_caption", "").strip()
        subfolder = "_".join(key.split("_")[:4])
        full_video_path = os.path.join(video_dir, subfolder, key) if not os.path.isabs(key) else key

        if not caption:
            null_records.append((idx, record, "no_caption"))
        elif not os.path.exists(full_video_path):
            null_records.append((idx, record, "missing_video"))
        else:
            score_records.append((idx, record))

    print(f"  {len(null_records)} records skipped (no caption / missing video)")
    print(f"  {len(score_records)} records to score ({args.num_frames} frames each)")

    processed = 0

    with open(events_jsonl_path, "a") as ef:
        for idx, record, reason in null_records:
            key = record["video_path"]
            out_idx = output_lookup[key]
            output_data[out_idx]["blip_itm_score"] = None
            output_data[out_idx]["blip_itm_reason"] = reason
            ef.write(json.dumps({"key": key, "blip_itm_score": None, "reason": reason}) + "\n")
            processed += 1
        ef.flush()

        for i, (idx, record) in enumerate(tqdm(score_records, desc="Scoring videos")):
            key = record["video_path"]
            caption = record["touch_caption"].strip()
            subfolder = "_".join(key.split("_")[:4])
            full_video_path = os.path.join(video_dir, subfolder, key) if not os.path.isabs(key) else key

            try:
                result = score_video(
                    model, processor, device,
                    full_video_path, caption,
                    num_frames=args.num_frames,
                    frame_batch_size=args.batch_size,
                )
                out_idx = output_lookup[key]
                output_data[out_idx]["blip_itm_score"] = result["mean"]
                output_data[out_idx]["blip_itm_score_max"] = result["max"]
                output_data[out_idx]["blip_itm_score_min"] = result["min"]
                output_data[out_idx]["blip_itm_num_frames"] = result["num_frames_scored"]
                ef.write(json.dumps({
                    "key": key,
                    "blip_itm_score": result["mean"],
                    "blip_itm_score_max": result["max"],
                    "blip_itm_score_min": result["min"],
                    "num_frames": result["num_frames_scored"],
                }) + "\n")
            except Exception as e:
                err_str = str(e)
                out_idx = output_lookup[key]
                output_data[out_idx]["blip_itm_score"] = None
                output_data[out_idx]["blip_itm_reason"] = f"error: {err_str}"
                ef.write(json.dumps({"key": key, "blip_itm_score": None, "error": err_str}) + "\n")

            ef.flush()
            processed += 1

            if args.snapshot_every > 0 and (i + 1) % args.snapshot_every == 0:
                write_json(output_json_path, output_data)

    write_json(output_json_path, output_data)

    print("\nDone.")
    print(f"Scored JSON: {os.path.abspath(output_json_path)}")
    print(f"Event log: {os.path.abspath(events_jsonl_path)}")
    print(f"Records processed: {processed}")


if __name__ == "__main__":
    main()
