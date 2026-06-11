"""
Gemini video annotation.

Walks a directory of clips, extracts 3 frames per clip, sends them to a Gemini
multimodal model along with the prompt in `gemini/prompt.txt`, and appends the
parsed JSON response to OUTPUT_FILE as JSONL. Safe to resume — already-processed
clips are skipped based on what's already in OUTPUT_FILE.
"""

import os
import time
import json
import logging
import multiprocessing
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type


# ===================== CONFIGURATION =====================
# API SETUP — set GOOGLE_API_KEY in your environment, or override here.
API_KEY     = os.environ.get("GOOGLE_API_KEY", "<YOUR_GOOGLE_API_KEY>")
MODEL_NAME  = "gemini-2.5-flash"  # flash is recommended for high-volume jobs

# PATHS
CLIPS_DIR     = Path("<path/to/clips>")              # directory of .mp4 clips (recursed)
OUTPUT_FILE   = Path("<path/to/gemini_labels.jsonl") # appended; used for resume
FRAMES_DIR    = Path("<path/to/frame_cache>")        # extracted frames are written here
LOG_FILE      = Path("<path/to/gemini.log>")
PROMPT_FILE   = Path(__file__).with_name("prompt.txt")

# PERFORMANCE
NUM_PROCESSES  = 16   # keep low (1–2) for free-tier API quotas
SAVE_INTERVAL  = 100  # flush results to disk after this many successes
MAX_RETRIES    = 10   # retries per clip before giving up
# =========================================================


logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


def configure_genai():
    genai.configure(api_key=API_KEY)


def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_representative_frames(video_path):
    """Extract frames at 25/50/75% of the clip, cache them, and return PIL Images."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 3:
        cap.release()
        raise ValueError(f"Video {video_path} has too few frames ({total_frames})")

    idxs = [total_frames // 4, total_frames // 2, (3 * total_frames) // 4]
    pil_images = []
    stem = Path(video_path).stem

    for i, frame_idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Could not read frame {frame_idx} from {video_path}")
            continue

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(FRAMES_DIR / f"{stem}_frame_{i}.jpg")
        pil_images.append(img)

    cap.release()

    if len(pil_images) < 3:
        raise ValueError(f"Failed to extract enough frames from {video_path}")
    return pil_images


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(Exception),
)
def generate_content_with_retry(model, clip_path, prompt):
    try:
        images = load_representative_frames(clip_path)
        response = model.generate_content([prompt, *images], safety_settings=SAFETY_SETTINGS)

        if not response.candidates or not response.candidates[0].content.parts:
            block_msg = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown Block"
            logging.warning(f"Content blocked for {clip_path}. Reason: {block_msg}")
            return {"text": None, "safety_ratings": f"Blocked: {block_msg}"}

        return {"text": response.text, "safety_ratings": None}

    except Exception as e:
        if "quick accessor" in str(e) or "candidates" in str(e):
            logging.error(f"Response empty for {clip_path}: {e}")
            return {"text": None, "safety_ratings": "Empty Response/Blocked"}
        if "429" in str(e) or "ResourceExhausted" in str(e):
            raise
        logging.error(f"Error processing {clip_path}: {e}")
        raise


def process_single_clip(video_path):
    configure_genai()
    model = genai.GenerativeModel(MODEL_NAME)
    clip_name = os.path.basename(video_path)
    prompt = load_prompt()

    try:
        result = generate_content_with_retry(model, video_path, prompt)
        return {
            "status":        "success",
            "clip":          clip_name,
            "response":      result["text"],
            "safety_reason": result["safety_ratings"],
            "timestamp":     datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status":    "failed",
            "clip":      clip_name,
            "error":     str(e),
            "timestamp": datetime.now().isoformat(),
        }


def load_processed_clips(output_file):
    """Read existing JSONL to skip clips that already have a successful entry."""
    processed = set()
    if not output_file.exists():
        return processed
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("status") == "success":
                    processed.add(data["clip"])
            except json.JSONDecodeError:
                continue
    return processed


def main():
    start_time = time.time()
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_clips = [str(p) for p in CLIPS_DIR.rglob("*") if p.is_file()]
    processed = load_processed_clips(OUTPUT_FILE)
    todo      = [c for c in all_clips if os.path.basename(c) not in processed]

    print(f"Found {len(all_clips)} clips. Skipping {len(processed)}. Processing {len(todo)}.")
    if not todo:
        print("All clips already processed.")
        return

    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        buffer = []
        print(f"Starting with {NUM_PROCESSES} processes...")

        for result in pool.imap_unordered(process_single_clip, todo):
            buffer.append(result)
            if len(buffer) >= SAVE_INTERVAL:
                print(f"--- Saving batch of {len(buffer)} results ---")
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    for res in buffer:
                        f.write(json.dumps(res) + "\n")
                buffer = []

        if buffer:
            print(f"--- Saving final {len(buffer)} results ---")
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for res in buffer:
                    f.write(json.dumps(res) + "\n")

    print(f"Done! Total time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
