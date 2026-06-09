import json
import argparse
import time
import numpy as np
from tqdm import tqdm
from clip_filter import load_model, compute_similarity, DEVICE

MODELS = [
    {"name": "EVA02-E-14-plus",            "pretrained": "laion2b_s9b_b144k",  "params": "5.0B"},
    {"name": "EVA02-E-14",                 "pretrained": "laion2b_s4b_b115k",  "params": "4.7B"},
    {"name": "ViT-bigG-14",                "pretrained": "laion2b_s39b_b160k", "params": "2.5B"},
    {"name": "ViT-H-14-378-quickgelu",     "pretrained": "dfn5b",              "params": "987M"},
    {"name": "ViT-H-14-quickgelu",         "pretrained": "dfn5b",              "params": "986M"},
]


def run_model(model_info, data, cache_dir=None):
    name = model_info["name"]
    pretrained = model_info["pretrained"]

    print(f"\n{'='*60}")
    print(f"  {name} ({pretrained})")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        model, preprocess, tokenizer = load_model(name, pretrained, cache_dir=cache_dir)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None
    load_time = time.time() - t0

    correct = 0
    total = 0
    errors = 0
    first_error = None
    true_scores = []
    best_wrong_scores = []
    margins = []

    t1 = time.time()
    for item in tqdm(data, desc=f"  Evaluating"):
        try:
            scores = compute_similarity(model, preprocess, tokenizer, item["image_path"], item["captions"])
        except Exception as e:
            errors += 1
            if first_error is None:
                first_error = f"{e} (image: {item['image_path']})"
            continue

        ti = item["true_caption_index"]
        pred = int(np.argmax(scores))

        true_score = scores[ti]
        wrong_scores = [scores[i] for i in range(len(scores)) if i != ti]
        best_wrong = max(wrong_scores)
        margin = true_score - best_wrong

        true_scores.append(true_score)
        best_wrong_scores.append(best_wrong)
        margins.append(margin)

        if pred == ti:
            correct += 1
        total += 1

    eval_time = time.time() - t1

    if errors > 0:
        print(f"  {errors}/{errors+total} images failed!")
        print(f"  First error: {first_error}")

    if total == 0:
        print(f"  ALL images failed - skipping this model.")
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    import torch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "pretrained": pretrained,
        "params": model_info["params"],
        "accuracy": correct / total * 100,
        "correct": correct,
        "total": total,
        "avg_true_score": float(np.mean(true_scores)),
        "avg_wrong_score": float(np.mean(best_wrong_scores)),
        "avg_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "positive_margin_pct": sum(1 for m in margins if m > 0) / len(margins) * 100,
        "load_time": load_time,
        "eval_time": eval_time,
        "speed": total / eval_time if eval_time > 0 else 0,
    }


def print_report(results):
    results = sorted(results, key=lambda r: r["accuracy"], reverse=True)

    W = 80
    print(f"\n{'='*W}")
    print(f"{'BENCHMARK REPORT':^{W}}")
    print(f"{'='*W}")

    print(f"\n{'-'*W}")
    print(f"  {'ACCURACY RANKING':^{W-4}}")
    print(f"{'-'*W}")
    print(f"  {'#':<3} {'Model':<30} {'Params':<8} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'-'*3} {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
    for i, r in enumerate(results):
        print(f"  {i+1}  {r['name']:<30} {r['params']:<8} {r['accuracy']:>9.1f}% {r['correct']:>4}/{r['total']}")

    print(f"\n{'-'*W}")
    print(f"  {'SCORE ANALYSIS':^{W-4}}")
    print(f"{'-'*W}")
    print(f"  {'Model':<30} {'Avg True':>10} {'Avg Wrong':>10} {'Avg Margin':>11} {'Med Margin':>11}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*11} {'-'*11}")
    for r in results:
        print(f"  {r['name']:<30} {r['avg_true_score']:>10.4f} {r['avg_wrong_score']:>10.4f} {r['avg_margin']:>+11.4f} {r['median_margin']:>+11.4f}")

    print(f"\n{'-'*W}")
    print(f"  {'SPEED & EFFICIENCY':^{W-4}}")
    print(f"{'-'*W}")
    print(f"  {'Model':<30} {'Params':<8} {'Load Time':>10} {'Eval Time':>10} {'Speed':>12}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
    for r in results:
        print(f"  {r['name']:<30} {r['params']:<8} {r['load_time']:>9.1f}s {r['eval_time']:>9.1f}s {r['speed']:>8.1f} img/s")

    print(f"\n{'='*W}")
    print(f"  {'RECOMMENDATION':^{W-4}}")
    print(f"{'='*W}")

    best_acc = results[0]
    best_margin = max(results, key=lambda r: r["avg_margin"])
    best_speed = max(results, key=lambda r: r["speed"])

    print(f"\n  Best Accuracy:     {best_acc['name']} ({best_acc['accuracy']:.1f}%)")
    print(f"  Best Margin:       {best_margin['name']} (avg margin: {best_margin['avg_margin']:+.4f})")
    print(f"  Best Speed:        {best_speed['name']} ({best_speed['speed']:.1f} img/s)")

    top2 = results[:2]
    print(f"\n  TOP 2 RECOMMENDED:")
    print(f"    1. {top2[0]['name']} - {top2[0]['accuracy']:.1f}% accuracy")
    print(f"    2. {top2[1]['name']} - {top2[1]['accuracy']:.1f}% accuracy")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run benchmark and generate report")
    parser.add_argument("--data", type=str, default="benchmark_data.json")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--models", type=int, default=5)
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    print(f"Benchmark: {len(data)} images, {args.models} models")
    print(f"Device: {DEVICE}")

    results = []
    for m in MODELS[:args.models]:
        r = run_model(m, data, cache_dir=args.cache_dir)
        if r:
            results.append(r)

    if results:
        print_report(results)
    else:
        print("No models loaded successfully.")


if __name__ == "__main__":
    main()
