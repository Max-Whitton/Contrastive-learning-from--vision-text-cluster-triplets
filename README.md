# Contrastive Learning from Vision-Text Cluster Triplets

![Overview](figures/athroughd-2-1.jpg)

This repo contains the code from TODO.

## Setup

Create the conda environment from `environment.yml` (env name `touch`):

```bash
conda env create -f environment.yml
conda activate touch
```

To update an existing `touch` env after pulling changes, run
`conda env update -f environment.yml --prune`.

## Data

## Preprocessing
### Clustering
### Filtering
The `data_filtering/` folder contains scripts for scoring and filtering image-caption pairs before training. Supports BLIP, OpenCLIP, SigLIP, and Qwen models. See `data_filtering/README.md` for usage.

## Training
See `scripts/triplet_run.sh` and 'scripts/speech/only_run.sh' for full examples with all hyperparameters.

![Attention maps](figures/attention_maps-1.png)

## Evaluation


## Structure

```
├── train.py              # main training script
├── run.sh                # example training command
├── data_filtering/       # scripts for filtering image-caption pairs
└── multimodal/           # model and data module code (submodule)
```


## Acknowledgment

This project builds on code from https://github.com/wkvong/multimodal-baby
