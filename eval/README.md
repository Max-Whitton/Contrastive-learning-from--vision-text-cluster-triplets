# Evaluation

`pv_eval.py` trains a small classifier head on top of a frozen ViT backbone and
reports accuracy on a 4-way "pick the image that matches the noun" task. The
same script is used for both Picture-Vocabulary (PV) and labeled-set categorical
eval — they differ only in which JSON datasets you pass.

Each JSON is a list of items shaped like:

```json
{
  "image": ["path/img0.png", "path/img1.png", "path/img2.png", "path/img3.png"],
  "conversations": [
    {"value": "Touch the image of 'apple'"},
    {"value": "B"}
  ]
}
```

## Usage

```bash
python eval/pv_eval.py \
    --backbone vit \
    --backbone_path models/touch_full.ckpt \
    --vocab_path multimodal/vocab.json \
    --train_path data/pv/train.json \
    --val_path   data/pv/val.json \
    --test_path  data/pv/test.json \
    --pool pad --variant Y --text_encoder own
```

For labeled-set categorical eval, point `--train_path/--val_path/--test_path`
at the labeled-set JSONs instead.

## Options

- `--backbone {vit,cvcl,touchv1,dino,clip_official,random}` — `vit` is the
  default and expects a checkpoint produced by this repo's `train.py`.
- `--variant {X,Y}` — X uses raw backbone features; Y attaches the pretrained
  projection head (frozen, for backbones that have one) or a fresh 1024→512
  linear (DINO/random).
- `--pool {pad,transformer}` — flatten the 4 per-image features into the MLP
  head, or run a small temporal transformer over them.
- `--text_encoder {clip,own}` — `own` reads the text-encoder embedding from the
  backbone checkpoint and needs `--vocab_path`. Falls back to CLIP for OOV
  nouns.
- `--exp TAG` — wandb group + filename prefix for checkpoints and feature caches.

Checkpoints land in `eval/checkpoints/` and image-feature caches in
`eval/feature_caches/`.
