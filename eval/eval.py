"""
Picture-vocabulary / labeled-set evaluation.

Picks which of 4 images matches a target noun on top of a frozen ViT backbone.
Two modes:
  - linear_probe: train a single nn.Linear head on the train split, pick best by val.
  - zero-shot:    no training; argmax cosine sim between (projected) image
                  features and the noun embedding.
The same script runs both the Picture-Vocabulary (PV) and "labeled-s"
categorical evals — pick which one with `--dataset {pv,labeled-s}`. JSON paths
inside `data/jsons/` are selected automatically; image paths inside those JSONs
are stored relative to the repo root and resolved at load time.

Example:
    python eval/eval.py \\
        --dataset pv \\
        --backbone_path models/touch_full.ckpt \\
        --mode linear_probe --variant Y --text_encoder own
"""

import argparse
import json
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tvm
import timm
import clip
import wandb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Standard ImageNet normalisation (non-CLIP backbones)
_IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
_CKPT_DIR   = os.path.join(_SCRIPT_DIR, "checkpoints")
_CACHE_DIR  = os.path.join(_SCRIPT_DIR, "feature_caches")
_JSONS_DIR  = os.path.join(_REPO_ROOT, "data", "jsons")

# Built-in dataset presets. Each maps to (train, val, test) JSON filenames
# under data/jsons/. The JSONs themselves hold image paths relative to the
# repo root (e.g. "data/pv/train/heart/heart_prob_000000.png").
_DATASETS = {
    "pv":        ("pv_train.json",        "pv_val.json",        "pv_test.json"),
    "labeled-s": ("labeled-s_train.json", "labeled-s_val.json", "labeled-s_test.json"),
}

TEXT_DIM    = 512   # CLIP and 'own' word embeddings are both 512-dim
NUM_IMAGES  = 4     # exactly 4 images per item
NUM_CLASSES = 4     # A / B / C / D
LABEL_MAP   = {"A": 0, "B": 1, "C": 2, "D": 3}


def _resolve(path):
    """Resolve a JSON-stored image path against the repo root."""
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


# ---------- noun helpers ----------
def _extract_noun(question):
    match = re.search(r"Touch the image of '(.+?)'", question)
    return match.group(1) if match else None


def _collect_nouns(json_paths):
    nouns = set()
    for path in json_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data:
            noun = _extract_noun(item["conversations"][0]["value"])
            if noun:
                nouns.add(noun)
    return sorted(nouns)


def _build_noun_embed_dict(text_encoder, backbone, json_paths, vocab_path=None, ckpt=None):
    """Return {noun: 512-d tensor} for nouns we can embed.
    Nouns we can't embed are excluded; callers (PVDataset) skip those items."""
    nouns = _collect_nouns(json_paths)
    print(f"[text] encoding {len(nouns)} unique nouns with encoder='{text_encoder}'")

    if text_encoder == 'clip':
        clip_model, _ = clip.load("ViT-B/16", device="cpu")
        clip_model.eval()
        tokens = clip.tokenize(nouns, truncate=True)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens).float()
        feats = F.normalize(feats, dim=-1)
        return {n: feats[i] for i, n in enumerate(nouns)}

    if not vocab_path:
        raise ValueError("--vocab_path is required when --text_encoder=own")
    with open(vocab_path) as f:
        vocab = json.load(f)
    embed_weight = ckpt["state_dict"]["text_encoder.embedding.weight"]  # [V, 512]

    oov_nouns, embed_dict = [], {}
    for noun in nouns:
        for key in [noun, noun.replace("the ", "", 1).strip(), noun.split()[0]]:
            if key in vocab:
                embed_dict[noun] = embed_weight[vocab[key]].clone().float()
                break
        else:
            oov_nouns.append(noun)

    if oov_nouns:
        print(f"  OOV: skipping {len(oov_nouns)}/{len(nouns)} nouns")
        print(f"  OOV nouns: {oov_nouns}")

    return embed_dict


# ---------- feature cache ----------
def build_feature_cache(vit, transform, device, backbone, variant, te,
                        exp, json_paths, noun_embed_dict):
    """Cache backbone features for every image that appears in an in-vocab item.
    Items whose noun is missing from `noun_embed_dict` are skipped, so we don't
    waste compute on images that will never be used downstream."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    job_id = os.environ.get("JOB_ID", str(os.getpid()))
    exp_tag = f"{exp}_" if exp else ""
    cache_path = os.path.join(
        _CACHE_DIR,
        f"cache_eval_{exp_tag}{backbone}_{variant}_{te}_{job_id}.pt",
    )
    all_paths = set()
    for json_path in json_paths:
        with open(json_path) as f:
            data = json.load(f)
        for item in data:
            noun = _extract_noun(item["conversations"][0]["value"])
            if noun not in noun_embed_dict:
                continue
            for p in item["image"]:
                all_paths.add(p)
    all_paths = sorted(all_paths)
    print(f"[cache] {len(all_paths)} unique images (in-vocab items only)")
    print(f"[cache] computing {backbone}/{variant} features...")

    cache = {}
    vit.eval()
    total = len(all_paths)
    log_every = max(1, total // 20)  # ~20 progress lines
    with torch.no_grad():
        for i in range(0, total, 256):
            batch = all_paths[i:i + 256]
            imgs = torch.stack([
                transform(Image.open(_resolve(p)).convert("RGB")) for p in batch
            ]).to(device)
            feats = vit(imgs).cpu()
            for p, f in zip(batch, feats):
                cache[p] = f
            if i // 256 % max(1, log_every // 256) == 0 or i + 256 >= total:
                print(f"[cache] {min(i + 256, total)}/{total} images", flush=True)
    torch.save(cache, cache_path)
    print(f"[cache] saved {len(cache)} features to {cache_path}")
    return cache


# ---------- dataset ----------
class PVDataset(Dataset):
    def __init__(self, json_path, transform=None, noun_embed_dict=None, feature_cache=None):
        with open(json_path) as f:
            raw = json.load(f)
        # Drop items whose noun isn't in noun_embed_dict — no fallback.
        self.data = [
            item for item in raw
            if _extract_noun(item["conversations"][0]["value"]) in noun_embed_dict
        ]
        skipped = len(raw) - len(self.data)
        if skipped:
            print(f"[data] {os.path.basename(json_path)}: skipped {skipped}/{len(raw)} OOV items")
        self.noun_embed_dict = noun_embed_dict
        self.transform = transform if transform is not None else _IMAGENET_TRANSFORM
        self.feature_cache = feature_cache

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = LABEL_MAP[item["conversations"][1]["value"]]
        noun = _extract_noun(item["conversations"][0]["value"])
        if self.feature_cache is not None:
            feats = torch.stack([self.feature_cache[p] for p in item["image"]])  # (4, D)
            return feats, label, self.noun_embed_dict[noun]
        images = torch.stack([
            self.transform(Image.open(_resolve(p)).convert("RGB"))
            for p in item["image"]
        ])  # (4, C, H, W)
        return images, label, self.noun_embed_dict[noun]


# ---------- heads ----------
class LinearProbeHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.layers = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.layers(x)


def fixed_sincos_pos_embed(n, d):
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------- eval loops ----------
def _batch_feats(vit, imgs, feat_proj):
    """Returns (B, N, D) feature tensor for either raw images or cached features."""
    if imgs.ndim == 5:                       # raw images (B, N, C, H, W)
        B, N, C, H, W = imgs.shape
        feats = vit(imgs.reshape(B * N, C, H, W))
    else:                                    # cached (B, N, D)
        B, N = imgs.shape[0], imgs.shape[1]
        feats = imgs.reshape(-1, imgs.shape[-1])
    feats = F.normalize(feats, dim=-1)
    if feat_proj is not None:
        feats = feat_proj(feats)
    return feats.reshape(B, N, -1)


def evaluate_linear_probe(vit, head, loader, device, feat_proj=None):
    head.eval()
    if feat_proj is not None:
        feat_proj.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels, nouns in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = _batch_feats(vit, imgs, feat_proj)
            B = feats.shape[0]
            noun_feats = F.normalize(nouns.to(device), dim=-1)
            logits = head(torch.cat([feats.reshape(B, -1), noun_feats], dim=1))
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def evaluate_zero_shot(vit, loader, device, feat_proj=None):
    """Pick the image whose (projected, normalized) feature has the highest
    cosine similarity to the noun embedding. No training, no head."""
    if feat_proj is not None:
        feat_proj.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels, nouns in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = _batch_feats(vit, imgs, feat_proj)         # (B, N, D)
            feats = F.normalize(feats, dim=-1)
            noun_feats = F.normalize(nouns.to(device), dim=-1) # (B, D)
            sims = (feats * noun_feats.unsqueeze(1)).sum(-1)   # (B, N)
            correct += (sims.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ---------- backbone loaders ----------
def _load_vit_checkpoint(backbone_path, variant):
    """Load a ViT-Large backbone trained with this repo's `vision_encoder.model.*` keys."""
    ckpt = torch.load(backbone_path, map_location="cpu")
    prefix = "vision_encoder.model."
    sd_full = ckpt["state_dict"]
    backbone_sd = {k[len(prefix):]: v for k, v in sd_full.items()
                   if k.startswith(prefix) and not k[len(prefix):].startswith("head.")}
    vit = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
    vit.load_state_dict(backbone_sd, strict=True)
    embedding_dim = 1024
    feat_proj = None
    if variant == 'Y':
        head_sd = {k[len(prefix + "head."):]: v for k, v in sd_full.items()
                   if k.startswith(prefix + "head.")}
        out_d, in_d = head_sd['weight'].shape
        feat_proj = nn.Linear(in_d, out_d, bias='bias' in head_sd)
        feat_proj.load_state_dict(head_sd)
        feat_proj.requires_grad_(False)
    print(f"[vit] {backbone_path} loaded {len(backbone_sd)} backbone keys OK")
    return vit, embedding_dim, feat_proj, ckpt


def _load_cvcl_checkpoint(backbone_path, variant):
    """Load original CVCL (resnext50) backbone."""
    ckpt = torch.load(backbone_path, map_location="cpu")
    prefix = "vision_encoder.model."
    sd_full = ckpt["state_dict"]
    backbone_sd = {k[len(prefix):]: v for k, v in sd_full.items()
                   if k.startswith(prefix) and not k[len(prefix):].startswith("fc.")}
    vit = tvm.resnext50_32x4d(pretrained=False)
    vit.fc = nn.Identity()
    vit.load_state_dict(backbone_sd, strict=True)
    embedding_dim = 2048
    feat_proj = None
    if variant == 'Y':
        fc_sd = {k[len(prefix + "fc."):]: v for k, v in sd_full.items()
                 if k.startswith(prefix + "fc.")}
        out_d, in_d = fc_sd['weight'].shape
        feat_proj = nn.Linear(in_d, out_d, bias='bias' in fc_sd)
        feat_proj.load_state_dict(fc_sd)
        feat_proj.requires_grad_(False)
    print(f"[cvcl] loaded {len(backbone_sd)} backbone keys OK")
    return vit, embedding_dim, feat_proj, ckpt


def _build_backbone(backbone, backbone_path, variant):
    """Returns (vit, embedding_dim, feat_proj, ckpt_or_None, transform_or_None)."""
    if backbone in ("vit", "touchv1"):
        if not backbone_path:
            raise ValueError(f"--backbone_path is required for --backbone={backbone}")
        vit, dim, feat_proj, ckpt = _load_vit_checkpoint(backbone_path, variant)
        return vit, dim, feat_proj, ckpt, None

    if backbone == "cvcl":
        if not backbone_path:
            raise ValueError("--backbone_path is required for --backbone=cvcl")
        vit, dim, feat_proj, ckpt = _load_cvcl_checkpoint(backbone_path, variant)
        return vit, dim, feat_proj, ckpt, None

    if backbone == "dino":
        if not backbone_path:
            raise ValueError("--backbone_path is required for --backbone=dino")
        ckpt = torch.load(backbone_path, map_location="cpu")
        prefix = "backbone."
        vision_sd = {k[len(prefix):]: v for k, v in ckpt["teacher"].items()
                     if k.startswith(prefix)}
        vit = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
        vit.load_state_dict(vision_sd, strict=True)
        feat_proj = nn.Linear(1024, 512, bias=False) if variant == 'Y' else None
        print(f"[dino] loaded {len(vision_sd)} keys OK")
        return vit, 1024, feat_proj, None, None

    if backbone == "clip_official":
        clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
        vit = clip_model.visual
        if variant == 'X':
            vit.proj = None
            return vit, 768, None, None, preprocess
        return vit, 512, None, None, preprocess

    if backbone == "random":
        vit = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
        num_tokens = vit.pos_embed.shape[1]
        with torch.no_grad():
            vit.pos_embed.copy_(
                fixed_sincos_pos_embed(num_tokens, 1024).unsqueeze(0)
            )
        vit.pos_embed.requires_grad = False
        feat_proj = nn.Linear(1024, 512, bias=False) if variant == 'Y' else None
        return vit, 1024, feat_proj, None, None

    raise ValueError(f"unknown backbone: {backbone}")


def _resolve_dataset_paths(args):
    """Fill train/val/test paths from --dataset, or use the explicit overrides."""
    if args.dataset is not None:
        train_name, val_name, test_name = _DATASETS[args.dataset]
        args.train_path = os.path.join(_JSONS_DIR, train_name)
        args.val_path   = os.path.join(_JSONS_DIR, val_name)
        args.test_path  = os.path.join(_JSONS_DIR, test_name)
    missing = [n for n in ("train_path", "val_path", "test_path") if not getattr(args, n)]
    if missing:
        raise ValueError(f"Missing JSON paths: {missing}. Pass --dataset or --{missing[0]}/etc.")


# ---------- main ----------
def main(args):
    os.makedirs(_CKPT_DIR, exist_ok=True)
    _resolve_dataset_paths(args)
    run_id = os.environ.get("JOB_ID", str(os.getpid()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, variant, mode = args.backbone, args.variant, args.mode

    vit, embedding_dim, feat_proj, ckpt, clip_preprocess = _build_backbone(
        backbone, args.backbone_path, variant
    )
    transform = clip_preprocess if clip_preprocess is not None else _IMAGENET_TRANSFORM

    # text encoder
    if args.text_encoder == 'own' and ckpt is None:
        print(f"[warn] --text_encoder=own not supported for {backbone}; using 'clip'")
        args.text_encoder = 'clip'
    noun_embed_dict = _build_noun_embed_dict(
        args.text_encoder, backbone,
        json_paths=[args.train_path, args.val_path, args.test_path],
        vocab_path=args.vocab_path,
        ckpt=ckpt if args.text_encoder == 'own' else None,
    )

    # zero-shot needs image features in the text-embedding space (512-d).
    # For variant=Y this is the pretrained/fresh proj; for clip_official Y the
    # 512-d projection lives inside vit. Otherwise we can't compare directly.
    if mode == "zero-shot":
        proj_dim = 512 if (feat_proj is not None or embedding_dim == 512) else embedding_dim
        if proj_dim != 512:
            raise ValueError(
                f"--mode=zero-shot needs image features in 512-d space; "
                f"{backbone}/{variant} gives {embedding_dim}-d. Use --variant Y."
            )

    head = None
    if mode == "linear_probe":
        proj_out = 512 if variant == 'Y' else embedding_dim
        head = LinearProbeHead(NUM_IMAGES * proj_out + TEXT_DIM, NUM_CLASSES)
        head = head.to(device)

    vit = vit.to(device)
    if feat_proj is not None:
        feat_proj = feat_proj.to(device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False

    # data
    feature_cache = build_feature_cache(
        vit, transform, device, backbone, variant, args.text_encoder,
        args.exp, [args.train_path, args.val_path, args.test_path],
        noun_embed_dict,
    )
    val_ds  = PVDataset(args.val_path,  noun_embed_dict=noun_embed_dict, feature_cache=feature_cache)
    test_ds = PVDataset(args.test_path, noun_embed_dict=noun_embed_dict, feature_cache=feature_cache)
    val_loader  = DataLoader(val_ds,  batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    if args.wandb:
        wandb.init(project=args.wandb_project,
                   name=f"pv-{backbone}-{variant}-{mode}-{args.text_encoder}",
                   group=args.exp or None)
        wandb.config.update(vars(args))

    # ── zero-shot: no training, just eval ────────────────────────────────────
    if mode == "zero-shot":
        val_acc  = evaluate_zero_shot(vit, val_loader,  device, feat_proj=feat_proj)
        test_acc = evaluate_zero_shot(vit, test_loader, device, feat_proj=feat_proj)
        if args.wandb:
            wandb.log({"val_acc": val_acc, "test_acc": test_acc})
        print(f"Zero-shot val acc:  {val_acc:.4f}")
        print(f"Zero-shot test acc: {test_acc:.4f}")
        return

    # ── linear-probe head: train on train_path, track best by val ────────────
    train_ds = PVDataset(args.train_path, noun_embed_dict=noun_embed_dict, feature_cache=feature_cache)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    best_val = 0.0
    criterion = nn.CrossEntropyLoss()
    params = list(head.parameters())
    if feat_proj is not None:
        params += [p for p in feat_proj.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    for epoch in range(args.epochs):
        head.train()
        if feat_proj is not None:
            feat_proj.train()

        for imgs, labels, nouns in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = _batch_feats(vit, imgs, feat_proj)
            B = feats.shape[0]
            noun_feats = F.normalize(nouns.to(device), dim=-1)
            logits = head(torch.cat([feats.reshape(B, -1), noun_feats], dim=1))

            loss = criterion(logits, labels)
            train_acc = (logits.argmax(1) == labels).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.wandb:
                wandb.log({"train_loss": loss.item(), "train_acc": train_acc})

        val_acc = evaluate_linear_probe(vit, head, val_loader, device, feat_proj=feat_proj)
        if args.wandb:
            wandb.log({"epoch": epoch + 1, "val_acc": val_acc})
        print(f"Epoch {epoch + 1}: val acc = {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            save_dict = {
                "head": head.state_dict(),
                "config": {
                    "backbone":      backbone,
                    "variant":       variant,
                    "mode":          mode,
                    "text_encoder":  args.text_encoder,
                    "embedding_dim": embedding_dim,
                    "backbone_path": args.backbone_path,
                },
            }
            if feat_proj is not None:
                save_dict["feat_proj"] = feat_proj.state_dict()
            exp_tag = f"{args.exp}_" if args.exp else ""
            torch.save(
                save_dict,
                os.path.join(_CKPT_DIR, f"eval_{exp_tag}{backbone}_{variant}_{args.text_encoder}_{run_id}.pt"),
            )
            print(f"  Saved best checkpoint (val={best_val:.4f})")

        test_acc = evaluate_linear_probe(vit, head, test_loader, device, feat_proj=feat_proj)
        if args.wandb:
            wandb.log({"test_acc": test_acc})
        print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Picture-vocabulary / labeled-s evaluation")
    parser.add_argument("--backbone", type=str, default="vit",
                        choices=["vit", "cvcl", "touchv1", "dino", "clip_official", "random"],
                        help="vit: ViT-Large trained with this repo's training script; "
                             "cvcl: original CVCL resnext50; touchv1: legacy ViT format; "
                             "dino/clip_official/random: baselines.")
    parser.add_argument("--backbone_path", type=str, default=None,
                        help="Path to the backbone checkpoint (.ckpt)")
    parser.add_argument("--variant", type=str, default="X", choices=["X", "Y"],
                        help="X: raw backbone features; Y: with pretrained or fresh projection head. "
                             "Zero-shot requires Y.")
    parser.add_argument("--mode", type=str, default="linear_probe",
                        choices=["linear_probe", "zero-shot"],
                        help="linear_probe: train a single Linear head on train+val; "
                             "zero-shot: argmax cosine sim of (projected) image features vs noun.")
    parser.add_argument("--text_encoder", type=str, default="own",
                        choices=["clip", "own"])
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="vocab JSON (required for --text_encoder=own)")

    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(_DATASETS.keys()),
                        help="Built-in dataset preset; picks train/val/test JSONs from data/jsons/.")
    parser.add_argument("--train_path", type=str, default=None,
                        help="Override train JSON path (ignored if --dataset is set)")
    parser.add_argument("--val_path",   type=str, default=None,
                        help="Override val JSON path")
    parser.add_argument("--test_path",  type=str, default=None,
                        help="Override test JSON path")

    parser.add_argument("--lr",     type=float, default=1e-6)
    parser.add_argument("--epochs", type=int,   default=50)

    parser.add_argument("--exp", type=str, default="",
                        help="Experiment tag (wandb group, prefixes ckpt and cache filenames)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log to wandb")
    parser.add_argument("--wandb_project", type=str, default="pv-eval")

    main(parser.parse_args())
