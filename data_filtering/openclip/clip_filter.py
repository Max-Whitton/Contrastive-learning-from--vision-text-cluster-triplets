import torch
import open_clip
from PIL import Image
import json
import os
import argparse
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEFAULT_MODEL = 'ViT-H-14'
DEFAULT_PRETRAINED = 'laion2b_s32b_b79k'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED, device=None, cache_dir=None):
    if device is None:
        device = DEVICE
    print(f"Loading {model_name} ({pretrained}) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device, cache_dir=cache_dir
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def _path_looks_like_video(path):
    lower = path.lower()
    return any(lower.endswith(s) for s in _VIDEO_SUFFIXES)


def load_rgb_pil(path):
    err_parts = []

    if not _path_looks_like_video(path):
        try:
            im = Image.open(path)
            im.load()
            return im.convert("RGB")
        except Exception as e:
            err_parts.append(f"PIL: {e}")

    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("cv2 could not open file")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise ValueError("cv2 could not read a frame")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        err_parts.append(f"cv2: {e}")

    try:
        from torchvision.io import read_video

        video, _, _ = read_video(path, pts_unit="sec")
        if video.numel() == 0:
            raise ValueError("no frames in video")
        idx = max(0, int(video.shape[0] // 2))
        return Image.fromarray(video[idx].cpu().numpy())
    except Exception as e:
        err_parts.append(f"torchvision.io.read_video: {e}")

    raise RuntimeError(f"Could not load image/video {path!r} ({'; '.join(err_parts)})")


def compute_similarity(model, preprocess, tokenizer, image_path, captions, device=None):
    if device is None:
        device = DEVICE
    if isinstance(captions, str):
        captions = [captions]

    image = preprocess(load_rgb_pil(image_path)).unsqueeze(0).to(device)
    text = tokenizer(captions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

    return similarities.tolist() if len(captions) > 1 else [similarities.item()]


class SAYCamDataset(Dataset):
    def __init__(self, metadata_path, image_dir, preprocess):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.data = []
        self.tokenizer = None

        print(f"Loading metadata from {metadata_path}...")
        try:
            if metadata_path.endswith('.jsonl'):
                with open(metadata_path, 'r') as f:
                    self.data = [json.loads(line) for line in f]
            else:
                with open(metadata_path, 'r') as f:
                    self.data = json.load(f)

            if len(self.data) > 0:
                print("Sample entry:", self.data[0])
                if 'caption' not in self.data[0] and 'text' not in self.data[0]:
                    print("WARNING: Neither 'caption' nor 'text' keys found in metadata!")
        except Exception as e:
            print(f"Error loading metadata: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_id = item.get('image_id', str(idx))
        img_filename = item.get('filename', item.get('image_path', f"{img_id}.jpg"))
        text = item.get('caption', item.get('text', item.get('utterance', '')))

        image_path = os.path.join(self.image_dir, img_filename)

        try:
            image = self.preprocess(Image.open(image_path))
        except Exception as e:
            image = torch.zeros((3, 224, 224))

        text_tokens = self.tokenizer([text])[0]

        return {
            "id": str(img_id),
            "image": image,
            "text_tokens": text_tokens,
            "raw_text": text
        }


def run_test_mode(model, preprocess, tokenizer):
    print("\n--- TEST MODE ---")

    test_img_path = "/projectnb/ivc-ml/ac25/BabyFM/Dataset/pretraining_dataset_raw/image/S_20140112_1426_04_180680_185280_frame_2.jpg"
    captions = [
        "In a Book Reading Setting. Turning a page a Book with your hand.",
        "Being touched on an unknown object on your hand.",
        "Interacting with your own body "
    ]

    try:
        scores = compute_similarity(model, preprocess, tokenizer, test_img_path, captions)

        print(f"\nImage: {test_img_path}")
        for i, caption in enumerate(captions):
            print(f"Score: {scores[i]:.4f} -> {caption}")

    except Exception as e:
        print(f"Test failed: {e}")


def run_batch_mode(model, preprocess, tokenizer, args):
    print("\n--- BATCH MODE ---")

    dataset = SAYCamDataset(args.metadata, args.image_dir, preprocess)
    dataset.tokenizer = tokenizer
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    output_filename = "openclip_scores.jsonl"
    print(f"Processing {len(dataset)} items...")
    print(f"Streaming results to {output_filename}...")

    with open(output_filename, 'w') as f:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Filtering"):
                images = batch["image"].to(DEVICE)
                text_tokens = batch["text_tokens"].to(DEVICE)

                image_features = model.encode_image(images)
                text_features = model.encode_text(text_tokens)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity_scores = (image_features * text_features).sum(dim=1)

                for i, score in enumerate(similarity_scores):
                    record = {
                        "image_id": batch["id"][i],
                        "score": score.item(),
                        "caption": batch["raw_text"][i]
                    }
                    f.write(json.dumps(record) + "\n")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCLIP Data Filtering")
    parser.add_argument('--mode', type=str, choices=['test', 'batch'], required=True)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--pretrained', type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument('--metadata', type=str, help="Path to metadata JSON")
    parser.add_argument('--image_dir', type=str, help="Directory containing images")
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    model, preprocess, tokenizer = load_model(args.model, args.pretrained)

    if args.mode == 'test':
        run_test_mode(model, preprocess, tokenizer)
    elif args.mode == 'batch':
        if not args.metadata or not args.image_dir:
            print("Error: --metadata and --image_dir required for batch mode.")
            sys.exit(1)
        run_batch_mode(model, preprocess, tokenizer, args)
