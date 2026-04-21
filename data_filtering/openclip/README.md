# OpenCLIP Data Filtering & Benchmarking

Filter image-caption pairs by cosine similarity and benchmark OpenCLIP models on caption-matching accuracy.

## Scripts

- `clip_filter.py` - Core filtering: computes cosine similarity between images and captions
- `score_labels.py` - Score a JSON file of labeled data
- `benchmark.py` - Evaluate models on caption-matching task
- `run_benchmark.py` - Run all 5 models and generate comparison report
- `prepare_benchmark.py` - Convert human labels to benchmark format
- `analyze_threshold.py` - Analyze score distribution for threshold selection

## Usage

```bash
# Score a dataset
python score_labels.py --input labels.json --output scored.json

# Analyze scores to pick a threshold
python analyze_threshold.py --input scored.json

# Prepare benchmark data from human labels
python prepare_benchmark.py --input human_labels.json --output benchmark_data.json

# Run benchmark on all models
python run_benchmark.py --data benchmark_data.json

# Quick benchmark with top 3 models only
python benchmark.py --data benchmark_data.json --models 3
```

## Benchmark input format

```json
[
  {
    "image_path": "/path/to/image.jpg",
    "captions": ["true caption", "distractor 1", "distractor 2"],
    "true_caption_index": 0
  }
]
```

## Requirements

```
torch
open_clip_torch
Pillow
tqdm
numpy
```
