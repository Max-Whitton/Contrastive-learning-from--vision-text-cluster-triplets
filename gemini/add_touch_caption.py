"""
Convert Gemini JSONL labels into a touch-caption JSON.

Reads the JSONL produced by `gemini/annotate.py`, keeps entries whose
mutually_exclusive label is "Touch (default)", builds a natural-language
caption from the structured checkbox/text fields, and writes a list of
{video_path, touch_caption} items to OUTPUT_JSON.
"""

import json
from pathlib import Path


# ===================== CONFIGURATION =====================
LABELS_JSONL = Path("<path/to/gemini_labels.jsonl>")
OUTPUT_JSON  = Path("<path/to/touch_captions.json>")
# =========================================================


def construct_caption(d):
    ans = ""
    for flag in d.get("checkbox_q1", {}):
        ans += flag + ". "
        if "Setting" in flag:
            ans = "In a " + ans

    found_verb = False
    for verb in d.get("checkbox_q2", {}):
        if d["checkbox_q2"][verb] > 1:
            found_verb = True
            ans += verb + ", "
            if "Being touched" in verb:
                ans += "on "
    for verb in d.get("text_q2", {}):
        ans += verb + ", "
        found_verb = True
    if not found_verb:
        ans += "Interacting with "
    else:
        ans = ans[:-2] + " "

    found_noun = False
    for noun in d.get("checkbox_q3", {}):
        if noun == "unknown":
            continue
        found_noun = True
        if noun != "Your own body":
            ans += " " + noun
        else:
            ans += "your own body"
        ans += ", "
    for noun in d.get("text_q3", {}):
        found_noun = True
        if noun != "Your own body":
            ans += " "
        ans += noun + ", "
    if not found_noun:
        ans += " an unknown object "
    else:
        ans = ans[:-2] + " "

    for body in d.get("checkbox_q4"):
        if d["checkbox_q4"][body] > 1:
            if "Being touched" in ans or "your own body" in ans:
                ans += "on your "
            else:
                ans += "with your "
            ans += body + ", "
    ans = ans[:-2] + "."
    ans = ans.replace("Turning a page", "Turning a page of")
    return ans.replace(" (default)", "")


def main():
    results = []
    with open(LABELS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    print(f"Loaded {len(results)} annotations")

    captions = []
    for q in results:
        try:
            if q["status"] != "success" or q["response"] is None:
                continue

            identifier = q["clip"][:-3] + "mp4"
            annotations = json.loads(q["response"])

            d = {}
            for category, res in annotations.items():
                d.setdefault(category, {})
                if not isinstance(res, list):
                    res = [res]
                for label in res:
                    d[category][label] = 3

            if "Touch" in d["mutually_exclusive"] or "Touch (default)" in d["mutually_exclusive"]:
                captions.append({
                    "video_path":    identifier,
                    "touch_caption": construct_caption(d),
                })
        except Exception:
            continue

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=4)
    print(f"Wrote {len(captions)} touch captions to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
