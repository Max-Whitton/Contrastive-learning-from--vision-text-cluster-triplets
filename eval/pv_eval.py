"""
Picture-vocabulary / labeled-set evaluation.

Trains a small classifier head (MLP or temporal transformer) on top of a frozen
ViT backbone to pick which of 4 images matches a target noun. The same script
runs both the Picture-Vocabulary (PV) and "labeled-s" categorical evals — they
differ only in which JSON datasets you point `--train_path`, `--val_path`, and
`--test_path` at.

Example:
    python eval/pv_eval.py \\
        --backbone_path models/touch_full.ckpt \\
        --train_path data/pv/train.json \\
        --val_path   data/pv/val.json \\
        --test_path  data/pv/test.json \\
        --pool pad --variant Y --text_encoder own
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
_CKPT_DIR   = os.path.join(_SCRIPT_DIR, "checkpoints")
_CACHE_DIR  = os.path.join(_SCRIPT_DIR, "feature_caches")

TEXT_DIM    = 512   # CLIP and 'own' word embeddings are both 512-dim
NUM_IMAGES  = 4     # exactly 4 images per item
NUM_CLASSES = 4     # A / B / C / D
LABEL_MAP   = {"A": 0, "B": 1, "C": 2, "D": 3}


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
    nouns = _collect_nouns(json_paths)
    print(f"[text] encoding {len(nouns)} unique nouns with encoder='{text_encoder}'")

    clip_model, _ = clip.load("ViT-B/16", device="cpu")
    clip_model.eval()

    def _clip_encode(noun_list):
        tokens = clip.tokenize(noun_list, truncate=True)
        with torch.no_grad():
            feats = clip_model.encode_text(tokens).float()
        return F.normalize(feats, dim=-1)

    if text_encoder == 'clip':
        feats = _clip_encode(nouns)
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
        print(f"  OOV ({len(oov_nouns)}): {oov_nouns} -> CLIP fallback")
        oov_feats = _clip_encode(oov_nouns)
        for i, n in enumerate(oov_nouns):
            embed_dict[n] = oov_feats[i]

    return embed_dict


# ---------- feature cache ----------
def build_feature_cache(vit, transform, device, backbone, variant, pool, te,
                        exp, json_paths):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    job_id = os.environ.get("JOB_ID", str(os.getpid()))
    exp_tag = f"{exp}_" if exp else ""
    cache_path = os.path.join(
        _CACHE_DIR,
        f"cache_eval_{exp_tag}{backbone}_{variant}_{pool}_{te}_{job_id}.pt",
    )
    all_paths = set()
    for json_path in json_paths:
        with open(json_path) as f:
            data = json.load(f)
        for item in data:
            for p in item["image"]:
                all_paths.add(p)
    all_paths = sorted(all_paths)
    print(f"[cache] computing {backbone}/{variant} features for {len(all_paths)} images...")

    cache = {}
    vit.eval()
    with torch.no_grad():
        for i in range(0, len(all_paths), 256):
            batch = all_paths[i:i + 256]
            imgs = torch.stack([
                transform(Image.open(p).convert("RGB")) for p in batch
            ]).to(device)
            feats = vit(imgs).cpu()
            for p, f in zip(batch, feats):
                cache[p] = f
    torch.save(cache, cache_path)
    print(f"[cache] saved {len(cache)} features to {cache_path}")
    return cache


# ---------- dataset ----------
class PVDataset(Dataset):
    def __init__(self, json_path, transform=None, noun_embed_dict=None, feature_cache=None):
        with open(json_path) as f:
            self.data = json.load(f)
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
            self.transform(Image.open(p).convert("RGB"))
            for p in item["image"]
        ])  # (4, C, H, W)
        return images, label, self.noun_embed_dict[noun]


# ---------- heads ----------
class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, layers):
        super().__init__()
        if layers == 3:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 1024), nn.ReLU(),
                nn.Linear(1024, 256),   nn.ReLU(),
                nn.Linear(256, num_classes),
            )
        elif layers == 2:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, 512), nn.ReLU(),
                nn.Linear(512, num_classes),
            )
        else:
            self.layers = nn.Sequential(nn.Linear(in_dim, num_classes))

    def forward(self, x):
        return self.layers(x)


class TemporalTransformer(nn.Module):
    """Small transformer encoder over per-frame CLS tokens."""

    def __init__(self, embed_dim, num_classes, num_layers=2, num_heads=4,
                 ff_dim=1024, max_len=16, extra_dim=0, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 2, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.noun_proj = nn.Linear(extra_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, extra_features=None):
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        if extra_features is not None and self.extra_dim > 0:
            noun_tok = self.noun_proj(extra_features).unsqueeze(1)
            x = torch.cat([cls, noun_tok, x], dim=1)
            x = x + self.pos_embed[:, :N + 2, :]
        else:
            x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embed[:, :N + 1, :]
        x = self.encoder(x)
        return self.classifier(self.norm(x[:, 0]))


def fixed_sincos_pos_embed(n, d):
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------- eval loop ----------
def evaluate(vit, head, loader, device, pool, feat_proj=None):
    head.eval()
    if feat_proj is not None:
        feat_proj.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels, nouns in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if imgs.ndim == 5:                       # raw images (B, N, C, H, W)
                B, N, C, H, W = imgs.shape
                feats = vit(imgs.reshape(B * N, C, H, W))
            else:                                    # cached (B, N, D)
                B, N = imgs.shape[0], imgs.shape[1]
                feats = imgs.reshape(-1, imgs.shape[-1])
            feats = F.normalize(feats, dim=-1)
            if feat_proj is not None:
                feats = feat_proj(feats)
            feats = feats.reshape(B, N, -1)
            noun_feats = F.normalize(nouns.to(device), dim=-1)
            if pool == "transformer":
                logits = head(feats, extra_features=noun_feats)
            else:
                logits = head(torch.cat([feats.reshape(B, -1), noun_feats], dim=1))
            correct += (logits.argmax(1) == labels).sum().item()
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


# ---------- main ----------
def main(args):
    os.makedirs(_CKPT_DIR, exist_ok=True)
    run_id = os.environ.get("JOB_ID", str(os.getpid()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone, pool, variant = args.backbone, args.pool, args.variant

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

    # head
    proj_out = 512 if variant == 'Y' else embedding_dim
    if pool == "transformer":
        head = TemporalTransformer(
            embed_dim=512, num_classes=NUM_CLASSES,
            num_layers=2, num_heads=4, ff_dim=1024,
            max_len=NUM_IMAGES, extra_dim=TEXT_DIM, dropout=0.1,
        )
        if feat_proj is None and embedding_dim != 512:
            feat_proj = nn.Linear(embedding_dim, 512)
    else:
        mlp_in = NUM_IMAGES * proj_out + TEXT_DIM
        head = MLPHead(mlp_in, NUM_CLASSES, args.layers)

    vit, head = vit.to(device), head.to(device)
    if feat_proj is not None:
        feat_proj = feat_proj.to(device)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad = False

    # data
    feature_cache = build_feature_cache(
        vit, transform, device, backbone, variant, pool, args.text_encoder,
        args.exp, [args.train_path, args.val_path, args.test_path],
    )
    train_ds = PVDataset(args.train_path, noun_embed_dict=noun_embed_dict, feature_cache=feature_cache)
    val_ds   = PVDataset(args.val_path,   noun_embed_dict=noun_embed_dict, feature_cache=feature_cache)
    test_ds  = PVDataset(args.test_path,  noun_embed_dict=noun_embed_dict, feature_cache=feature_cache)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128)
    test_loader  = DataLoader(test_ds,  batch_size=128)

    # training
    best_val = 0.0
    criterion = nn.CrossEntropyLoss()
    params = list(head.parameters())
    if feat_proj is not None:
        params += [p for p in feat_proj.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    if args.wandb:
        wandb.init(project=args.wandb_project,
                   name=f"pv-{backbone}-{variant}-{pool}-{args.text_encoder}",
                   group=args.exp or None)
        wandb.config.update(vars(args))

    for epoch in range(args.epochs):
        head.train()
        if feat_proj is not None:
            feat_proj.train()

        for imgs, labels, nouns in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if imgs.ndim == 5:
                B, N, C, H, W = imgs.shape
                with torch.no_grad():
                    feats = vit(imgs.reshape(B * N, C, H, W))
            else:
                B, N = imgs.shape[0], imgs.shape[1]
                feats = imgs.reshape(-1, imgs.shape[-1])
            feats = F.normalize(feats, dim=-1)
            if feat_proj is not None:
                feats = feat_proj(feats)
            feats = feats.reshape(B, N, -1)

            noun_feats = F.normalize(nouns.to(device), dim=-1)
            if pool == "transformer":
                logits = head(feats, extra_features=noun_feats)
            else:
                logits = head(torch.cat([feats.reshape(B, -1), noun_feats], dim=1))

            loss = criterion(logits, labels)
            train_acc = (logits.argmax(1) == labels).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.wandb:
                wandb.log({"train_loss": loss.item(), "train_acc": train_acc})

        val_acc = evaluate(vit, head, val_loader, device, pool, feat_proj=feat_proj)
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
                    "pool":          pool,
                    "text_encoder":  args.text_encoder,
                    "layers":        args.layers,
                    "embedding_dim": embedding_dim,
                    "backbone_path": args.backbone_path,
                },
            }
            if feat_proj is not None:
                save_dict["feat_proj"] = feat_proj.state_dict()
            exp_tag = f"{args.exp}_" if args.exp else ""
            torch.save(
                save_dict,
                os.path.join(_CKPT_DIR, f"pv_eval_{exp_tag}{backbone}_{variant}_{pool}_{args.text_encoder}_{run_id}.pt"),
            )
            print(f"  Saved best checkpoint (val={best_val:.4f})")

        test_acc = evaluate(vit, head, test_loader, device, pool, feat_proj=feat_proj)
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
                        help="X: raw backbone features + MLP; Y: with pretrained or fresh projection head")
    parser.add_argument("--pool", type=str, default="pad",
                        choices=["pad", "transformer"])
    parser.add_argument("--text_encoder", type=str, default="own",
                        choices=["clip", "own"])
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="vocab JSON (required for --text_encoder=own)")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path",   type=str, required=True)
    parser.add_argument("--test_path",  type=str, required=True)

    parser.add_argument("--lr",     type=float, default=1e-6)
    parser.add_argument("--epochs", type=int,   default=50)
    parser.add_argument("--layers", type=int,   default=2)

    parser.add_argument("--exp", type=str, default="",
                        help="Experiment tag (wandb group, prefixes ckpt and cache filenames)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log to wandb")
    parser.add_argument("--wandb_project", type=str, default="pv-eval")

    main(parser.parse_args())
