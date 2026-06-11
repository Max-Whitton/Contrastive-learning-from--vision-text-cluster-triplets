# Contrastive Learning from Vision-Text Cluster Triplets
This repo contains the code from TODO.

**A)** *Examples of [`frame, speech transcript, touch ID`] triplets (top) and data
pairs missing either speech or touch (bottom). **B)** Visualization of model
architecture and contrastive learning. **C)** Labeled-S test sample.
**D)** PV test sample.*


![Overview](figures/athroughd-2-1.jpg)




## Setup

Create the conda environment from `environment.yml` (env name `touch`):

```bash
conda env create -f environment.yml
conda activate touch
```

## Data
### To use our data:
Download `clips`, `jsons`, `labeled-s`, and `pv` folders from (Databrary link TODO), and place them in `data/clips`, `data/jsons`, `data/labeled-s`, and `data/pv`.

### To use your own:
Drop your raw video clips under `data/clips/` and your training JSON(s) under
`data/jsons/`. Each training JSON is a list of items shaped like:

```json
{"video_path": "data/clips/<clip>.mp4", "audio_caption": "...", "touch_caption": "..."}
```

`touch_caption` is optional but required for the triplet loss. Then run the
preprocessing steps below to score/filter the pairs and add
`touch_cluster_{4,16,64,256}` keys for clustering-based training.

For the picture-vocabulary and labeled-S evals, drop your frames under
`data/pv/` or `data/labeled-s/` and add JSONs at `data/jsons/pv_{train,val,test}.json`
and `data/jsons/labeled-s_{train,val,test}.json`. Each item must follow the
4-image / Q+A schema documented in `eval/README.md`.

## Preprocessing
These steps can be skipped if using our prepared data.
### Filtering
The `data_filtering/` folder contains scripts for scoring and filtering image-caption pairs before training. Supports BLIP, OpenCLIP, SigLIP, and Qwen models. See `data_filtering/README.md` for usage.

### Gemini touch labels
Touch captions can be auto-generated with Gemini in two steps. Defaults assume
the repo layout (`data/clips/` for input, `gemini/` for intermediate output,
`data/jsons/` for the final captions), so if you keep that layout you only
need to set your API key:

```bash
export GOOGLE_API_KEY=...
```

1. **Annotate clips** with `gemini/annotate.py`. The defaults read clips from
   `data/clips/`, write labels to `gemini/gemini_labels.jsonl`, cache
   extracted frames in `gemini/frame_cache/`, and log to `gemini/gemini.log`.
   Override any of these in the `CONFIGURATION` block at the top of the file
   if your paths differ, then run:
   ```bash
   python gemini/annotate.py
   ```
   The prompt lives in `gemini/prompt.txt`; tweak it there. The script is
   resumable — re-running skips clips already in `OUTPUT_FILE`.
2. **Build natural-language captions** with `gemini/add_touch_caption.py`.
   Defaults read `gemini/gemini_labels.jsonl` and write
   `data/jsons/touch_captions.json` (a list of
   `{"video_path", "touch_caption"}` items ready to merge into your training
   JSON). Override `LABELS_JSONL` / `OUTPUT_JSON` at the top of the file if
   needed, then run:
   ```bash
   python gemini/add_touch_caption.py
   ```

### Clustering
Set `path` and `CLUSTER_NUMBERS` in `run_k_means.py`, then run this file. Note that larger k values in `CLUSTER_NUMBERS` take a very long time to run.

## Training
See `scripts/triplet_run.sh` and `scripts/speech/only_run.sh` for full examples with all hyperparameters.

![Attention maps](figures/attention_maps-1.png)

## Evaluation
This repo supports eval on Labeled-S and Picture Vocabulary. See `eval/README.md` for details on `eval/eval.py` configurations.


## Structure

```
├── train.py              # main training script
├── run.sh                # example training command
├── data_filtering/       # scripts for filtering image-caption pairs
└── multimodal/           # model and data module code (submodule)
```


## Acknowledgment

This project builds on code from https://github.com/wkvong/multimodal-baby
