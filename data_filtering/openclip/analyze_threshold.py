import json
import argparse
import numpy as np


def _iter_entries(data):
    if isinstance(data, dict):
        return data.items()
    if isinstance(data, list):
        return enumerate(data)
    raise TypeError(f"JSON root must be dict or list, got {type(data).__name__}")


def analyze(input_path):
    with open(input_path) as f:
        data = json.load(f)

    valid_scores = []
    rejected = 0
    errors = 0

    for _, entry in _iter_entries(data):
        if not isinstance(entry, dict):
            continue
        score = entry.get("clip_score", None)
        if score is None:
            continue
        elif score == -1:
            rejected += 1
        elif score == -2:
            errors += 1
        else:
            valid_scores.append(score)

    valid_scores = np.array(valid_scores)

    print(f"\n{'='*60}")
    print(f"  SCORE DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    print(f"\n  Total entries:  {len(data)}")
    print(f"  Rejected:       {rejected}")
    print(f"  Errors:         {errors}")
    print(f"  Valid scores:   {len(valid_scores)}")

    if len(valid_scores) == 0:
        print("  No valid scores to analyze!")
        return

    print(f"\n{'-'*60}")
    print(f"  STATISTICS")
    print(f"{'-'*60}")
    print(f"  Min:    {valid_scores.min():.4f}")
    print(f"  Max:    {valid_scores.max():.4f}")
    print(f"  Mean:   {valid_scores.mean():.4f}")
    print(f"  Median: {np.median(valid_scores):.4f}")
    print(f"  Std:    {valid_scores.std():.4f}")

    print(f"\n{'-'*60}")
    print(f"  PERCENTILES")
    print(f"{'-'*60}")
    for p in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]:
        val = np.percentile(valid_scores, p)
        print(f"  {p:>3}th percentile: {val:.4f}")

    print(f"\n{'-'*60}")
    print(f"  THRESHOLD SIMULATION")
    print(f"  (What % of data you'd KEEP at each threshold)")
    print(f"{'-'*60}")
    print(f"  {'Threshold':>10} {'Keep':>8} {'Remove':>8} {'Keep %':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for t in np.arange(0.10, 0.45, 0.02):
        keep = (valid_scores >= t).sum()
        remove = (valid_scores < t).sum()
        pct = keep / len(valid_scores) * 100
        print(f"  {t:>10.2f} {keep:>8} {remove:>8} {pct:>7.1f}%")

    print(f"\n{'-'*60}")
    print(f"  HISTOGRAM")
    print(f"{'-'*60}")
    bins = np.arange(0.05, 0.55, 0.02)
    hist, edges = np.histogram(valid_scores, bins=bins)
    max_count = max(hist)
    bar_width = 40

    for i in range(len(hist)):
        bar_len = int(hist[i] / max_count * bar_width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {edges[i]:.2f}-{edges[i+1]:.2f} | {bar:<{bar_width}} {hist[i]}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(valid_scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        axes[0].axvline(valid_scores.mean(), color='red', linestyle='--', label=f'Mean: {valid_scores.mean():.3f}')
        axes[0].axvline(np.median(valid_scores), color='orange', linestyle='--', label=f'Median: {np.median(valid_scores):.3f}')
        axes[0].set_xlabel('CLIP Cosine Similarity Score')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Score Distribution')
        axes[0].legend()

        sorted_scores = np.sort(valid_scores)
        cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1].plot(sorted_scores, 1 - cdf, color='steelblue', linewidth=2)
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Fraction of Data KEPT')
        axes[1].set_title('How much data you keep at each threshold')
        axes[1].grid(True, alpha=0.3)

        for t in [0.20, 0.25, 0.30]:
            kept = (valid_scores >= t).mean()
            axes[1].axvline(t, color='red', linestyle=':', alpha=0.5)
            axes[1].annotate(f't={t:.2f}\nkeep {kept:.0%}',
                           xy=(t, kept), fontsize=8,
                           textcoords="offset points", xytext=(10, 0))

        plt.tight_layout()
        plot_path = input_path.replace('.json', '_distribution.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\n  Plot saved to: {plot_path}")

    except ImportError:
        print("\n  (Install matplotlib for visual plots: pip install matplotlib)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze score distribution for threshold selection")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    analyze(args.input)
