# Data Filtering

Scripts for filtering image-caption pairs using different vision-language models. Each model computes a score for how well a caption matches its image, which can then be used to filter out low-quality pairs.

## Structure

```
data_filtering/
├── blip/          # BLIP ITM scoring (good for video data)
├── openclip/      # OpenCLIP scoring + benchmarking tools
├── siglip/        # SigLIP 2 scoring (sigmoid-based)
└── qwen/          # Qwen2-VL yes/no verification
```

## Requirements

```bash
pip install torch transformers open_clip_torch pillow tqdm opencv-python numpy
```

## Usage

### OpenCLIP (recommended for images)

Score a JSON file with image-caption pairs:
```bash
cd openclip
python score_labels.py --input labels.json --output scored.json
```

Analyze the score distribution to pick a threshold:
```bash
python analyze_threshold.py --input scored.json
```

Benchmark different CLIP models on your data:
```bash
python benchmark.py --data benchmark_data.json --models 3
```

### BLIP (good for video data)

Scores video-caption pairs by extracting multiple frames:
```bash
cd blip
python compute_score.py --input_json videos.json --video_dir /path/to/videos --num_frames 4
```

### SigLIP

Similar to OpenCLIP but uses sigmoid scoring:
```bash
cd siglip
python compute_score.py --input_json labels.json --batch_size 64
```

### Qwen

Uses a VLM to directly ask "Does this caption describe the image?" (YES/NO):
```python
from qwen.qwen_filter import load_qwen, compute_qwen_score

model, processor = load_qwen("Qwen/Qwen2-VL-2B-Instruct", "cuda")
score = compute_qwen_score(model, processor, "cuda", "image.jpg", "a dog playing")
# Returns 1.0 for YES, 0.0 for NO, 0.5 for uncertain
```

## Input Format

Most scripts expect a JSON file where each entry has:
- `frame_path` or `image_path`: path to the image
- `human_caption` or `gemini_caption`: the caption text

Example:
```json
{
  "sample_001": {
    "frame_path": "/data/images/001.jpg",
    "human_caption": "a child playing with blocks"
  }
}
```

## Output

All scorers add a score field to each entry (e.g., `clip_score`, `siglip_score`, `blip_itm_score`). Jobs are resumable - if interrupted, they pick up where they left off using the events JSONL log.
