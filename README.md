# UnSegGNetSTD

UnSegGNetSTD extends UnSegGNet with Soft Threshold Dynamics (STD) for
unsupervised binary image segmentation. Images are represented as patch graphs
using pretrained DINO features, then optimized per image with either the
original GCN softmax output or the STD-regularized output.

## Repository layout

- `main.py`: batch evaluation entry point.
- `train_fn.py`: reusable evaluation function used by parameter search.
- `StdParamSearch.py`: Optuna search for STD hyperparameters.
- `SegmentViz.py`: evaluation with predicted-mask export.
- `segment.py`: DINO feature extraction, per-image optimization, and metrics.
- `gnn_pool.py`: GCN, DMON/NCUT losses, and STD output implementation.
- `datasets/DATASETS.md`: dataset download and directory instructions.

## Setup

1. Create and activate a clean Python environment.
2. Install PyTorch and PyTorch Geometric with `./script.sh`.
3. Install the remaining dependencies with `pip install -r requirements.txt`.
4. Prepare CUB, ECSSD, or DUTS according to `datasets/DATASETS.md`.

The DINO checkpoint is downloaded automatically on first use.

## Unified initialization checkpoint

`model.pt` stores the common random initialization used for all images and for
the baseline/STD comparison. Shared GCN and MLP parameters are restored before
each image is optimized. STD-only scalar parameters use the values supplied by
`--alpha`, `--lam`, and `--eps` when they are absent from the checkpoint.

Create or replace the checkpoint by running an entry point with `--gen_init`,
or use `main.py`, which initializes it by default. Use `--checkpoint PATH` to
keep multiple initialization states.

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

## Hyperparameter search and visualizations

Run the 50-trial Optuna search with:

```bash
python StdParamSearch.py --dataset ECSSD --activation STD --gen_init
```

Generate masks using the saved best parameters with:

```bash
python SegmentViz.py --dataset ECSSD --activation STD
```

Search artifacts are stored under `search_results/`, and exported masks are
stored under `output/<dataset>/`.

## Supported datasets

The public evaluation path supports CUB, ECSSD, and DUTS. CVC is also accepted
by the loader when prepared locally. Dataset files are intentionally excluded
from Git.

## Acknowledgements and license

This repository extends the original UnSegGNet implementation and includes
code derived from DINO ViT feature extraction and DeepCut. See
`README_ORIGIN.md` for the original project description and acknowledgements.
The repository is distributed under the MIT License in `LICENSE`.
