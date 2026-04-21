import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def load_qwen(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
    )

    model.eval()
    return model, processor


@torch.no_grad()
def compute_qwen_score(
    model,
    processor,
    device: str,
    image_path: str,
    caption: str,
) -> float:

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (224, 224), (0, 0, 0))

    prompt = f"""
Does this caption correctly describe the image?
Caption: {caption}
Answer only YES or NO
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenizer=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=5,
    )

    response = processor.decode(
        output_ids[0], skip_special_tokens=True
    ).lower()

    if "yes" in response:
        return 1.0
    if "no" in response:
        return 0.0

    return 0.5


def run_test_qwen(
    device: str,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    test_img_path: str = "",
    threshold: float | None = 0.7,
):

    print("\nRunning Qwen test")

    captions = [
        "In a Book Reading Setting. Turning a page a Book with your hand.",
        "Being touched on an unknown object on your hand.",
        "Interacting with your own body",
        "Balloon floating in the sky",
        "Cars crashing on the highway",
    ]

    model, processor = load_qwen(model_id, device)

    print(f"\nImage: {test_img_path}")
    print(f"Threshold: {threshold}\n")

    for caption in captions:
        prob = compute_qwen_score(
            model, processor, device, test_img_path, caption
        )

        if threshold is None:
            print(f"Score: {prob:.4f} -> {caption}")
        else:
            decision = "KEEP" if prob >= threshold else "DROP"
            print(f"Score: {prob:.4f} [{decision}] -> {caption}")


def run_batch_for_qwen(
    dataloader,
    metadata_path: str,
    device: str,
    scores_output_path: str,
    filtered_metadata_output_path: str,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    threshold: float = 0.7,
):

    print("\n--Running Qwen in batch--")
    print(f"Checkpoint: {model_id}")
    print(f"Threshold: {threshold}")

    model, processor = load_qwen(model_id, device)

    if metadata_path.endswith(".jsonl"):
        with open(metadata_path, "r") as f:
            original_data = [json.loads(line) for line in f]
    else:
        with open(metadata_path, "r") as f:
            original_data = json.load(f)

    updated_data = list(original_data)

    kept = 0
    total = 0
    meta_idx = 0

    with open(scores_output_path, "w") as sf:
        for batch in tqdm(dataloader, desc="Filtering(Qwen-VL)"):
            image_paths = list(batch["image_path"])
            captions = list(batch["raw_text"])
            ids = list(batch["id"])

            bsz = len(ids)

            for i in range(bsz):
                prob_i = compute_qwen_score(
                    model,
                    processor,
                    device,
                    image_paths[i],
                    captions[i],
                )

                keep_i = prob_i >= threshold

                sf.write(
                    json.dumps(
                        {
                            "image_id": ids[i],
                            "qwen_score": prob_i,
                            "keep": keep_i,
                        }
                    )
                    + "\n"
                )

                if meta_idx < len(updated_data):
                    updated_data[meta_idx]["qwen_score"] = prob_i
                    updated_data[meta_idx]["keep"] = keep_i

                meta_idx += 1
                total += 1
                kept += int(keep_i)

    if filtered_metadata_output_path.endswith(".jsonl"):
        with open(filtered_metadata_output_path, "w") as f:
            for row in updated_data:
                f.write(json.dumps(row) + "\n")
    else:
        with open(filtered_metadata_output_path, "w") as f:
            json.dump(updated_data, f, indent=2)

    print("\n--- SUMMARY ---")
    print(f"Total processed: {total}")
    print(f"Kept: {kept}")
    print(f"Dropped: {total - kept}")
    print("Done!")
