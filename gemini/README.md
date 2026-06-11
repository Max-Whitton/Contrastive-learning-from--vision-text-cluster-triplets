# Gemini touch labels

Auto-generate touch captions for a directory of video clips with the Gemini
multimodal API. Two steps:

1. `annotate.py` — sample 3 frames per clip, send them to Gemini with the
   prompt in `prompt.txt`, and append the structured JSON response to a JSONL
   label file.
2. `add_touch_caption.py` — turn that JSONL into a list of
   `{"video_path", "touch_caption"}` items, dropping clips Gemini did not
   confidently label as "Touch".

Defaults assume the repo layout (`data/clips/` for input, `gemini/` for
intermediate output, `data/jsons/` for the final captions), so if you keep
that layout you only need to set your API key:

```bash
export GOOGLE_API_KEY=...
```

## 1. Annotate clips

`annotate.py` reads clips from `CLIPS_DIR` (default `data/clips/`), writes
labels to `OUTPUT_FILE` (default `gemini/gemini_labels.jsonl`), caches
extracted frames in `FRAMES_DIR` (default `gemini/frame_cache/`), and logs to
`LOG_FILE` (default `gemini/gemini.log`). Override any of these in the
`CONFIGURATION` block at the top of the file if your paths differ, then run:

```bash
python gemini/annotate.py
```

The prompt lives in `prompt.txt`; tweak it there. The script is resumable —
re-running skips clips already present in `OUTPUT_FILE`.

Other knobs in the `CONFIGURATION` block:

- `MODEL_NAME` — Gemini model id; defaults to `gemini-2.5-flash`, which is
  recommended for high-volume jobs.
- `NUM_PROCESSES` — multiprocessing pool size. Keep this low (1–2) if you're
  on the free-tier API quota.
- `SAVE_INTERVAL` — flush results to disk after this many successful labels.
- `MAX_RETRIES` — retries per clip before giving up (uses tenacity with
  exponential backoff).

## 2. Build natural-language captions

`add_touch_caption.py` reads `LABELS_JSONL` (default
`gemini/gemini_labels.jsonl`) and writes `OUTPUT_JSON` (default
`data/jsons/touch_captions.json`):

```bash
python gemini/add_touch_caption.py
```

Output is a list of `{"video_path", "touch_caption"}` items ready to merge
into your training JSON under `data/jsons/`. Items whose
`mutually_exclusive` label is anything other than `Touch (default)` are
dropped.
