import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import timm

from multimodal.multimodal_data_module import IMAGE_H, IMAGE_W

# =========================================================
# ARGPARSE
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser("ViT Attention Map Generator")

    parser.add_argument("--backbone_paths", nargs="+", required=True, default=["models/vit_speech_only.ckpt", "models/vit_triplet_256_clusters.ckpt"])
    parser.add_argument("--image_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="attention_outputs")

    return parser.parse_args()


args = parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOAD MODEL (same logic as yours)
# =========================================================
def load_vit(backbone_path):
    ckpt = torch.load(backbone_path, map_location="cpu")
    sd_full = ckpt["state_dict"]
    prefix = "vision_encoder.model."

    backbone_sd = {
        k[len(prefix):]: v
        for k, v in sd_full.items()
        if k.startswith(prefix) and not k[len(prefix):].startswith("head.")
    }

    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0
    )

    model.load_state_dict(backbone_sd, strict=True)
    model = model.to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


# =========================================================
# TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_H, IMAGE_W)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================================================
# ATTENTION EXTRACTION
# =========================================================
def get_attention_map(model, image_tensor):
    """
    Returns CLS attention map (averaged over heads)
    """

    attn = []

    def hook_fn(module, input, output):
        # output shape: (B, heads, tokens, tokens)
        attn.append(output.detach())

    # register hook on last block
    handle = model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(image_tensor)

    handle.remove()

    attn_map = attn[0]  # (1, heads, tokens, tokens)
    attn_map = attn_map.mean(dim=1)  # average heads → (1, tokens, tokens)

    # CLS token attention to patches
    cls_attn = attn_map[0, 0, 1:]  # ignore CLS-to-CLS

    num_patches = cls_attn.shape[0]
    size = int(np.sqrt(num_patches))

    cls_attn = cls_attn.reshape(size, size)

    cls_attn = cls_attn.cpu().numpy()
    cls_attn = cls_attn / cls_attn.max()

    return cls_attn


# =========================================================
# SAVE VISUALIZATION
# =========================================================
def save_attention(image, attn_map, save_path):
    attn_map = Image.fromarray((attn_map * 255).astype(np.uint8))
    attn_map = attn_map.resize(image.size, resample=Image.BILINEAR)

    attn_map = np.array(attn_map)

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(attn_map, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# MAIN LOOP
# =========================================================
for backbone_path in args.backbone_paths:
    print(f"Loading model: {backbone_path}")
    model = load_vit(backbone_path)

    backbone_name = os.path.basename(backbone_path).replace(".ckpt", "")
    out_dir = os.path.join(args.output_dir, backbone_name)
    os.makedirs(out_dir, exist_ok=True)

    for img_path in args.image_paths:
        if not os.path.exists(img_path):
            print(f"Missing: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(DEVICE)

        try:
            attn_map = get_attention_map(model, x)

            filename = os.path.basename(img_path)
            save_path = os.path.join(out_dir, filename)

            save_attention(image, attn_map, save_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")