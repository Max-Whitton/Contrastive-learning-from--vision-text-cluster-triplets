import json
import argparse
import random
import re


def extract_keywords(caption):
    stopwords = {
        'a', 'an', 'the', 'in', 'on', 'with', 'your', 'of', 'and', 'to',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'setting',
    }
    words = re.findall(r'[a-z]+', caption.lower())
    return set(w for w in words if w not in stopwords and len(w) > 1)


def word_overlap_ratio(caption1, caption2):
    kw1 = extract_keywords(caption1)
    kw2 = extract_keywords(caption2)
    if not kw1 or not kw2:
        return 0.0
    intersection = kw1 & kw2
    return len(intersection) / min(len(kw1), len(kw2))


def select_distractors(true_caption, all_captions, num_distractors=2, max_overlap=0.5):
    candidates = []
    for cap in all_captions:
        if cap == true_caption:
            continue
        overlap = word_overlap_ratio(true_caption, cap)
        candidates.append((cap, overlap))

    candidates.sort(key=lambda x: x[1])

    low_overlap = [c for c, o in candidates if o <= max_overlap]
    high_overlap = [c for c, o in candidates if o > max_overlap]

    distractors = []

    if len(low_overlap) >= num_distractors:
        distractors = random.sample(low_overlap, num_distractors)
    else:
        distractors = low_overlap[:]
        remaining = num_distractors - len(distractors)
        if high_overlap:
            distractors += random.sample(high_overlap, min(remaining, len(high_overlap)))

    return distractors


def prepare_benchmark_data(input_path, output_path, num_distractors=2, max_overlap=0.5, seed=42):
    random.seed(seed)

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} total entries from {input_path}")

    valid_entries = []
    rejected = 0
    for filename, info in data.items():
        caption = info["human_caption"].strip()
        if caption.lower() == "reject":
            rejected += 1
            continue
        valid_entries.append({
            "filename": filename,
            "caption": caption,
            "frame_path": info["frame_path"],
        })

    print(f"Valid entries (with captions): {len(valid_entries)}")
    print(f"Rejected entries: {rejected}")

    if len(valid_entries) < num_distractors + 1:
        print(f"Error: Need at least {num_distractors + 1} valid entries, got {len(valid_entries)}")
        return

    all_captions = list(set(entry["caption"] for entry in valid_entries))
    print(f"Unique captions: {len(all_captions)}")

    benchmark_data = []
    overlap_stats = []

    for entry in valid_entries:
        true_caption = entry["caption"]

        distractors = select_distractors(true_caption, all_captions, num_distractors, max_overlap)

        if len(distractors) < num_distractors:
            print(f"Warning: Could only find {len(distractors)} distractors for: {true_caption}")
            continue

        for d in distractors:
            overlap_stats.append(word_overlap_ratio(true_caption, d))

        captions = [true_caption] + distractors
        indices = list(range(len(captions)))
        random.shuffle(indices)
        shuffled_captions = [captions[i] for i in indices]
        true_caption_index = indices.index(0)

        benchmark_data.append({
            "image_path": entry["frame_path"],
            "captions": shuffled_captions,
            "true_caption_index": true_caption_index,
        })

    with open(output_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nSaved {len(benchmark_data)} benchmark items to {output_path}")

    if overlap_stats:
        avg_overlap = sum(overlap_stats) / len(overlap_stats)
        print(f"Avg keyword overlap between true & distractors: {avg_overlap:.2f}")
        print(f"Max overlap allowed: {max_overlap}")

    print("\n--- Sample entries ---")
    for item in benchmark_data[:3]:
        print(f"\nImage: {item['image_path']}")
        for i, cap in enumerate(item['captions']):
            true_marker = " <--" if i == item['true_caption_index'] else ""
            print(f"  [{i}] {cap}{true_marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare benchmark data from human labels")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="benchmark_data.json")
    parser.add_argument("--max_overlap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_benchmark_data(args.input, args.output, max_overlap=args.max_overlap, seed=args.seed)
