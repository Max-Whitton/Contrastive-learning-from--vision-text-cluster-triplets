# Evaluation

`eval.py` runs the 4-way "pick the image that matches the noun" task on top of
a frozen ViT backbone. It covers the Picture-Vocabulary (PV) and labeled-S
categorical evals, in two modes:

- **Linear probe** (`--mode linear_probe`) — train a single `nn.Linear` head on
  the train split, select the best epoch by val accuracy, report test accuracy.
- **Zero-shot** (`--mode zero-shot`) — no training; argmax cosine similarity
  between the (projected) image features and the noun embedding. Requires
  `--variant Y` so image and text features live in the same 512-d space.

Each JSON input is a list of items shaped like:

```json
{
  "image": ["path/img0.png", "path/img1.png", "path/img2.png", "path/img3.png"],
  "conversations": [
    {"value": "Touch the image of 'apple'"},
    {"value": "B"}
  ]
}
```

## Quick start

The dataset preset (`--dataset pv` or `--dataset labeled-s`) picks the
train/val/test JSONs from `data/jsons/` automatically:

```bash
# PV, linear probe
python eval/eval.py --dataset pv --mode linear_probe \
    --backbone vit --backbone_path models/touch_full.ckpt \
    --variant Y --text_encoder own --vocab_path multimodal/vocab.json

# labeled-S, linear probe
python eval/eval.py --dataset labeled-s --mode linear_probe \
    --backbone vit --backbone_path models/touch_full.ckpt \
    --variant Y --text_encoder own --vocab_path multimodal/vocab.json

# PV, zero-shot
python eval/eval.py --dataset pv --mode zero-shot --variant Y \
    --backbone vit --backbone_path models/touch_full.ckpt \
    --text_encoder own --vocab_path multimodal/vocab.json

# labeled-S, zero-shot
python eval/eval.py --dataset labeled-s --mode zero-shot --variant Y \
    --backbone vit --backbone_path models/touch_full.ckpt \
    --text_encoder own --vocab_path multimodal/vocab.json
```

To point at custom JSONs instead of a preset, drop `--dataset` and pass
`--train_path/--val_path/--test_path` directly.

A convenience wrapper with sensible defaults lives at
`scripts/pv_eval_run.sh`; override via env vars, e.g.

```bash
MODEL=models/touch_full.ckpt DATASET=labeled-s MODE=zero-shot \
    bash scripts/pv_eval_run.sh
```

## Options

- `--dataset {pv,labeled-s}` — built-in preset for train/val/test JSONs.
- `--mode {linear_probe,zero-shot}` — train a single Linear head vs. zero-shot
  cosine similarity.
- `--backbone {vit,cvcl,touchv1,dino,clip_official,random}` — `vit` is the
  default and expects a checkpoint produced by this repo's `train.py`.
- `--variant {X,Y}` — X uses raw backbone features; Y attaches the pretrained
  projection head (frozen, for backbones that have one) or a fresh 1024→512
  linear (DINO/random). Zero-shot requires Y.
- `--text_encoder {clip,own}` — `own` reads the text-encoder embedding from
  the backbone checkpoint and needs `--vocab_path`.
- `--exp TAG` — wandb group + filename prefix for checkpoints and feature caches.
- `--wandb` — log to wandb (off by default).

Checkpoints land in `eval/checkpoints/` and image-feature caches in
`eval/feature_caches/`.
