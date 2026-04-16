import argparse
import json
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.manifold import TSNE
from tqdm import tqdm
from PIL import Image
import sys
import timm

sys.path.append('/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby')
import multimodal
from multimodal.multimodal_data_module import (
    IMAGE_H, IMAGE_W
)

# =========================================================
# ARGPARSE
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE visualization of ViT embeddings")

    parser.add_argument("--json_path", type=str,
                        default="data/jsons/full_merged.json")

    parser.add_argument("--max_samples", type=int, default=100000)

    parser.add_argument("--backbone_path", type=str,
                        default="models/vit_triplet_256_clusters.ckpt")

    parser.add_argument("--output_path", type=str,
                        default=None)

    parser.add_argument("--perplexity", type=float, default=1)

    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--tsne_dim", type=int, default=2)

    parser.add_argument("--point_size", type=float, default=10)

    parser.add_argument("--cmap", type=str, default="tab20")

    parser.add_argument("--num_clusters", type=int, default=8)

    return parser.parse_args()


args = parse_args()

# ---- DEVICE ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOAD CHECKPOINT
# =========================================================
ckpt = torch.load(args.backbone_path, map_location="cpu")
sd_full = ckpt["state_dict"]
prefix = "vision_encoder.model."


# -------------------------
# BACKBONE WEIGHTS
# -------------------------
backbone_sd = {
    k[len(prefix):]: v
    for k, v in sd_full.items()
    if k.startswith(prefix) and not k[len(prefix):].startswith("head.")
}

vit = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)
vit.load_state_dict(backbone_sd, strict=True)
vit = vit.to(DEVICE).eval()

for p in vit.parameters():
    p.requires_grad = False


# -------------------------
# feat_proj
# -------------------------
head_sd = {
    k[len(prefix + "head."):]: v
    for k, v in sd_full.items()
    if k.startswith(prefix + "head.")
}

feat_proj = None
if len(head_sd) > 0:
    out_dim, in_dim = head_sd["weight"].shape

    feat_proj = torch.nn.Linear(in_dim, out_dim, bias="bias" in head_sd)
    feat_proj.load_state_dict(head_sd)
    feat_proj = feat_proj.to(DEVICE).eval()

    for p in feat_proj.parameters():
        p.requires_grad = False


# =========================================================
# HELPERS
# =========================================================
def get_center_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_embedding(image):
    image = Image.fromarray(image)
    image = image.resize((IMAGE_W, IMAGE_H))

    normalizer = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalizer,
    ])

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = vit(x)

        if feat_proj is not None:
            feats = feat_proj(feats)

        feats = torch.nn.functional.normalize(feats, dim=-1)

    return feats.cpu().numpy().squeeze()


# =========================================================
# LOAD DATA
# =========================================================
with open(args.json_path, "r") as f:
    data = json.load(f)

embeddings, labels = [], []


# =========================================================
# PROCESS
# =========================================================
for item in tqdm(data[:args.max_samples]):
    full_path = item.get("video_path")
    label = item.get(f"touch_cluster_{args.num_clusters}")

    if not full_path or label is None:
        continue
    if not os.path.exists(full_path):
        continue

    frame = get_center_frame(full_path)
    if frame is None:
        continue

    try:
        emb = extract_embedding(frame)
        embeddings.append(emb)
        labels.append(label)
    except Exception as e:
        print(e)
        continue


embeddings = np.array(embeddings)


# =========================================================
# TSNE
# =========================================================
tsne = TSNE(
    n_components=args.tsne_dim,
    perplexity=args.perplexity,
    random_state=args.random_state
)

reduced = tsne.fit_transform(embeddings)


# =========================================================
# COLOR MAPPING
# =========================================================
unique_labels = list(set(labels))
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
colors = [label_to_idx[l] for l in labels]


# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(10, 8))
plt.scatter(
    reduced[:, 0],
    reduced[:, 1],
    c=colors,
    cmap=args.cmap,
    s=args.point_size
)
unique_labels = sorted(set(labels))  # instead of list(set(...))
label_to_idx = {l: i for i, l in enumerate(unique_labels)}
colors = [label_to_idx[l] for l in labels]

cmap = plt.get_cmap(args.cmap)
norm = plt.Normalize(vmin=0, vmax=len(unique_labels) - 1)

handles = [
    plt.Line2D(
        [],
        [],
        marker='o',
        linestyle='',
        color=cmap(norm(label_to_idx[label])),
        label=str(label)
    )
    for label in unique_labels
]

plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("t-SNE of ViT + feat_proj Embeddings (Center Frames)")
plt.tight_layout()
if args.output_path:
    plt.savefig(args.output_path)
else:
    save_path = f"visualizations/output/{os.path.basename(args.backbone_path).split('.')[0]}_{args.num_clusters}_clusters_{args.max_samples}_points_{args.perplexity}_perplexity.pdf"
    plt.savefig(save_path)