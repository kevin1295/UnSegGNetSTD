# UnSegGNetSTD

UnSegGNetSTD extends UnSegGNet with Soft Threshold Dynamics (STD) for
unsupervised binary image segmentation. Images are represented as patch graphs
using pretrained DINO features, then optimized per image with either the
original GCN softmax output or the STD-regularized output.

This repository contains the reference implementation of the UnSegGNet + STD
pipeline. Citation metadata is in `CITATION.cff`, the full reproduction protocol
is in `REPRODUCIBILITY.md`, and upstream code provenance and licenses are
listed in `THIRD_PARTY_NOTICES.md`. Reference quantitative results are reported
in the paper and are intentionally not duplicated here.

## Repository layout

- `main.py`: batch evaluation entry point.
- `train_fn.py`: reusable evaluation function used by parameter search.
- `StdParamSearch.py`: Optuna search for STD hyperparameters.
- `SegmentViz.py`: evaluation with predicted-mask export.
- `segment.py`: DINO feature extraction, per-image optimization, and metrics.
- `gnn_pool.py`: GCN, DMON/NCUT losses, and STD output implementation.
- `model.pt`: the shared random initialization checkpoint (see below).
- `datasets/DATASETS.md`: dataset download and directory instructions.
- `REPRODUCIBILITY.md`: environment, commands, randomness, and failure modes.
- `THIRD_PARTY_NOTICES.md`: provenance and licenses of third-party code.
- `CITATION.cff`: how to cite this repository and the associated manuscript.

## Setup

1. Create and activate a clean Python environment.
2. Install PyTorch and PyTorch Geometric with `./script.sh`.
3. Install the remaining dependencies with `pip install -r requirements.txt`.
4. Prepare CUB, ECSSD, or DUTS according to `datasets/DATASETS.md`.

The DINO checkpoint is downloaded automatically on first use and is expected at
`dino_deitsmall8_pretrain_full_checkpoint.pth` in the repository root; if the
download is blocked, place the file there manually.

See `REPRODUCIBILITY.md` for the versions recorded by this repository, the full
command set, and known failure modes.

## Unified initialization checkpoint

`model.pt` (about 108 KB) is part of this repository on purpose. It stores the
single random initialization that is shared by every image and by both the
baseline and the STD run: the GCN and MLP parameters are restored from this
checkpoint before each image is optimized, so baseline and STD results differ
only in the output layer (softmax versus STD-regularized softmax). Because of
that, it is a reproduction prerequisite rather than a disposable artifact.
STD-only scalar parameters (`alpha`, `lam`, `eps`) are supplied by
`--alpha`, `--lam`, and `--eps` when they are absent from the checkpoint.

Regenerate it with `--gen_init`, which saves the freshly initialized state
before any optimization:

```bash
python SegmentViz.py --dataset ECSSD --activation STD --gen_init
```

`main.py` always regenerates the checkpoint at startup (its `gen_init` flag is
enabled unconditionally), so a plain `python main.py ...` run also rewrites
`model.pt`. With the default `--seed 42` the regenerated state is the same one
that ships in this repository.

Use `--checkpoint PATH` to keep several initialization states side by side.

## Evaluation

Baseline:

```bash
python main.py --dataset ECSSD --activation selu --threshold 0.3 --bs False
```

STD-GCN:

```bash
python main.py --dataset ECSSD --activation STD --threshold 0.3 \
  --alpha 2.0 --lam 1.0 --eps 3.0 --nb_iter 3 --bs False
```

Both commands use seed 42 by default. Override it with `--seed`. The final
summary reports successful and failed samples separately; a run with no valid
samples exits with an error instead of reporting an invalid mIoU.

The tuned values used in the manuscript were found with the Optuna search
described in `REPRODUCIBILITY.md`; the search artifacts themselves are not
published (see the note below).

## Hyperparameter search and visualizations

Run the 50-trial Optuna search with:

```bash
python StdParamSearch.py --dataset ECSSD --activation STD --gen_init
```

The search writes `search_results/optuna.db`, a per-dataset
`search_results/<dataset>/best_std_params.json`, and a set of Optuna plots
under `search_results/<dataset>/std_visualizations/`.

Generate masks using the saved best parameters with:

```bash
python SegmentViz.py --dataset ECSSD --activation STD
```

Exported masks are stored under `output/<dataset>/`.

**Note on artifacts that are not published.** `search_results/` (the Optuna
databases, best-parameter JSON files, and search plots), `optuna_analysis.py`
and `build_vis_grid.py` are author-internal experiment products and plotting
scripts. They are not distributed with this repository, and neither the search
results nor the figure-building scripts should be expected here. Only the code
needed to re-run the search from scratch is public.

Consequently `SegmentViz.py` cannot load hyperparameters out of the box: by
default it reads `search_results/<dataset>/best_std_params.json`. Either run
`StdParamSearch.py` first, or pass `--no_load_hparams` together with explicit
`--alpha/--lam/--eps/--nb_iter` values, e.g.

```bash
python SegmentViz.py --dataset ECSSD --activation STD --no_load_hparams \
  --alpha <alpha> --lam <lambda> --eps <epsilon> --nb_iter <T>
```

Fill in the tuned values from your own `StdParamSearch.py` run; the values used
for the manuscript are reported in the paper.

## MedSAM weights

The MedSAM baseline (`--process MEDSAM_INFERENCE`) loads weights through
`transformers` from the Hugging Face Hub repository
[`wanglab/medsam-vit-base`](https://huggingface.co/wanglab/medsam-vit-base),
which is downloaded automatically on first use. Pre-download it explicitly with:

```bash
python -c "from transformers import SamModel, SamProcessor; \
SamModel.from_pretrained('wanglab/medsam-vit-base'); \
SamProcessor.from_pretrained('wanglab/medsam-vit-base')"
```

For offline machines, set `HF_HOME` to a pre-populated cache directory, or
mirror the download with `HF_ENDPOINT`.

The standalone MedSAM package is pinned to a specific commit:

```bash
pip install "git+https://github.com/bowang-lab/MedSAM.git@d71e8a1a99ad751840a22a7fa3ecfb4166fb1488"
```

MedSAM weights are optional. Nothing in this repository loads `medsam_vit_b.pth`:
`features_extract.py` imports `sam_model_registry` at module level but never
builds a SAM model, so only the package (pinned above) has to be installed. The
previous `gdown` fallback that pulled the `.pth` from Google Drive was removed —
the link is dead and `gdown` is no longer a requirement.

## Supported datasets

The public evaluation path supports CUB, ECSSD, and DUTS. CVC is also accepted
by the loader when prepared locally. Dataset files are intentionally excluded
from Git.

## Acknowledgements and license

This repository extends the original UnSegGNet implementation and includes code
derived from DINO ViT feature extraction and DeepCut. See `README_ORIGIN.md`
for the original project description and acknowledgements, and
`THIRD_PARTY_NOTICES.md` for the per-component license summary.

The repository is distributed under the MIT License in `LICENSE`
(Copyright (c) 2024 Sai Gopal Reddy Kovvuri, inherited from the upstream
UnSegGNet project).
