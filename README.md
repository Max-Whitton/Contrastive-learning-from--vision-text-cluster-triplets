# Contrastive Learning from Vision-Text Cluster Triplets

Contrastive learning with vision, audio, and touch modalities using triplet data.

## Structure

```
├── train.py              # main training script
├── run.sh                # example training command
├── data_filtering/       # scripts for filtering image-caption pairs
└── multimodal/           # model and data module code (submodule)
```

## Setup

```bash
pip install torch pytorch_lightning transformers torchinfo wandb
```

## Training

```bash
python train.py \
  --triplet \
  --train_data_path data/training.json \
  --pretrained_ckpt models/pretrained.ckpt \
  --num_touch_classes 256 \
  --batch_size 64 \
  --max_epochs 2000
```

See `run.sh` for full example with all hyperparameters.

## Data Filtering

The `data_filtering/` folder contains scripts for scoring and filtering image-caption pairs before training. Supports BLIP, OpenCLIP, SigLIP, and Qwen models. See `data_filtering/README.md` for usage.

## Acknowledgment

This project builds on code from https://github.com/wkvong/multimodal-baby
