

import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from transformers import CLIPProcessor, CLIPModel
import json

# -----------------------------
# Configuration
# -----------------------------
path = "data/jsons/full_touch.json"
MODEL_NAME = "openai/clip-vit-large-patch14"  # powerful pre-trained CLIP
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set NUM_CLUSTERS=None to automatically find best k using silhouette score
# NUM_CLUSTERS = 16

# -----------------------------
# Example data
# -----------------------------
# Each dict has video_path and touch_caption
with open(path, "r") as f:
    data = json.load(f)

captions = [item["touch_caption"] for item in data]

# -----------------------------
# Load CLIP model
# -----------------------------
print("Loading CLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Use fp16 for efficiency
model = model.half()

# -----------------------------
# Encode captions
# -----------------------------
def encode_texts(texts, batch_size):
    all_embeddings = []

    if len(texts) == 0:
        raise ValueError("No captions provided!")

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding captions"):
            batch = texts[i:i+batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                output = model.get_text_features(**inputs)

            # Extract tensor safely
            if hasattr(output, "pooler_output"):
                features = output.pooler_output
            elif isinstance(output, torch.Tensor):
                features = output
            else:
                raise ValueError("Cannot extract tensor from get_text_features output")

            # Normalize
            features /= features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().numpy())

    return np.vstack(all_embeddings)
print("Encoding captions...")
embeddings = encode_texts(captions, BATCH_SIZE)


# -----------------------------
# Final clustering
# -----------------------------
for NUM_CLUSTERS in [4, 16, 64, 256]:
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=20, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # -----------------------------
    # Add cluster index to each dict
    # -----------------------------
    for item, cluster_id in zip(data, labels):
        item[f"touch_cluster_{NUM_CLUSTERS}"] = int(cluster_id)

    # -----------------------------
    # Print clusters
    # -----------------------------
    clusters = {}
    for item in data:
        clusters.setdefault(item[f"touch_cluster_{NUM_CLUSTERS}"], []).append(item["touch_caption"])


  
    path = "data/jsons/full_touch.json"

    with open(path, "w") as f:
        json.dump(data, f, indent=4)