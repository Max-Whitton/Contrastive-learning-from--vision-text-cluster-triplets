import torch
import json
import argparse
import time
import numpy as np
from tqdm import tqdm
from clip_filter import load_model, compute_similarity, DEVICE

MODELS = [
    {"name": "ViT-H-14-378-quickgelu", "pretrained": "dfn5b",             "note": "#1 avg perf (70.8%)"},
    {"name": "ViT-H-14-quickgelu",     "pretrained": "dfn5b",             "note": "#2 avg perf (69.6%)"},
    {"name": "EVA02-E-14-plus",        "pretrained": "laion2b_s9b_b144k", "note": "#3 avg perf (69.3%)"},
    {"name": "EVA02-E-14",              "pretrained": "laion2b_s4b_b115k", "note": "#4 avg perf (66.9%)"},
    {"name": "ViT-L-14-quickgelu",     "pretrained": "dfn2b",             "note": "#5 avg perf (66.9%)"},
]


def load_benchmark_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} benchmark items")

    for i, item in enumerate(data):
        assert "image_path" in item, f"Item {i} missing 'image_path'"
        assert "captions" in item, f"Item {i} missing 'captions'"
        assert "true_caption_index" in item, f"Item {i} missing 'true_caption_index'"
        assert len(item["captions"]) >= 2, f"Item {i} needs at least 2 captions"
        assert 0 <= item["true_caption_index"] < len(item["captions"]), \
            f"Item {i} has invalid true_caption_index={item['true_caption_index']}"

    return data


def benchmark_model(model_info, data, cache_dir=None):
    model_name = model_info["name"]
    pretrained = model_info["pretrained"]

    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({pretrained})")
    print(f"Note:  {model_info['note']}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        model, preprocess, tokenizer = load_model(model_name, pretrained, cache_dir=cache_dir)
    except Exception as e:
        print(f"  ERROR: Failed to load model: {e}")
        print(f"  Skipping {model_name}...")
        return None
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")

    correct = 0
    total = 0
    all_scores = []
    details = []

    for item in tqdm(data, desc=f"Evaluating {model_name}"):
        image_path = item["image_path"]
        captions = item["captions"]
        true_idx = item["true_caption_index"]

        try:
            scores = compute_similarity(model, preprocess, tokenizer, image_path, captions)
        except Exception as e:
            print(f"  Warning: Could not process {image_path}: {e}")
            continue

        predicted_idx = int(np.argmax(scores))
        is_correct = (predicted_idx == true_idx)
        correct += int(is_correct)
        total += 1
        all_scores.append(scores[true_idx])

        details.append({
            "image_path": image_path,
            "true_idx": true_idx,
            "predicted_idx": predicted_idx,
            "correct": is_correct,
            "scores": {cap: round(s, 4) for cap, s in zip(captions, scores)},
        })

    accuracy = (correct / total * 100) if total > 0 else 0.0
    eval_time = time.time() - start_time - load_time

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "pretrained": pretrained,
        "note": model_info["note"],
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_true_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "load_time_s": round(load_time, 1),
        "eval_time_s": round(eval_time, 1),
        "details": details,
    }


def print_summary(results):
    print(f"\n{'='*70}")
    print(f"{'BENCHMARK RESULTS':^70}")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Pretrained':<20} {'Accuracy':>10}")
    print(f"{'-'*35} {'-'*20} {'-'*10}")

    results_sorted = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    for r in results_sorted:
        print(f"{r['model']:<35} {r['pretrained']:<20} {r['accuracy']:>9.1f}%")

    print(f"{'-'*70}")

    best = results_sorted[0]
    worst = results_sorted[-1]
    print(f"\nBest:  {best['model']} ({best['pretrained']}) -> {best['accuracy']:.1f}%")
    print(f"Worst: {worst['model']} ({worst['pretrained']}) -> {worst['accuracy']:.1f}%")
    print(f"Gap:   {best['accuracy'] - worst['accuracy']:.1f} percentage points")

    print(f"\n{'='*70}")
    print(f"FAILURE ANALYSIS (Best model: {best['model']})")
    print(f"{'='*70}")
    failures = [d for d in best["details"] if not d["correct"]]
    if failures:
        for f in failures[:10]:
            print(f"\n  Image: {f['image_path']}")
            print(f"  True idx: {f['true_idx']}, Predicted idx: {f['predicted_idx']}")
            for cap, score in f["scores"].items():
                marker = " <-- TRUE" if list(f["scores"].keys()).index(cap) == f["true_idx"] else ""
                print(f"    {score:.4f} -> {cap}{marker}")
    else:
        print("  No failures! Perfect accuracy.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenCLIP models on caption matching")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--models", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    data = load_benchmark_data(args.data)
    models_to_test = MODELS[:args.models]

    print(f"\nBenchmarking {len(models_to_test)} models on {len(data)} images")
    print(f"Device: {DEVICE}")
    if args.cache_dir:
        print(f"Cache dir: {args.cache_dir}")

    results = []
    for model_info in models_to_test:
        result = benchmark_model(model_info, data, cache_dir=args.cache_dir)
        if result is None:
            print(f"\n  -> {model_info['name']}: SKIPPED (failed to load)")
            continue
        print(f"\n  -> {model_info['name']}: {result['accuracy']:.1f}% "
              f"({result['correct']}/{result['total']})")
        results.append(result)

    if results:
        print_summary(results)
    else:
        print("\nNo models were successfully loaded. Check disk space and model names.")

    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
