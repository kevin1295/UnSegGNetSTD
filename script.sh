#!/bin/bash
set -eu

pip install torch==2.4.0+cu124 torchvision==0.19.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu124

# Set the TORCH environment variable to the PyTorch version
TORCH=$(python -c "import torch; print(torch.__version__)")

# Export the TORCH environment variable
export TORCH

# Install the required Python packages
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install torch_geometric
pip install class_resolver
pip install -r requirements.txt

# MedSAM is pinned to a specific commit so that the reported results stay
# reproducible (this is the current HEAD of bowang-lab/MedSAM).
MEDSAM_COMMIT=d71e8a1a99ad751840a22a7fa3ecfb4166fb1488
pip install "git+https://github.com/bowang-lab/MedSAM.git@${MEDSAM_COMMIT}"

# ---------------------------------------------------------------------------
# MedSAM weights
#
# The evaluation code loads MedSAM through `transformers` from the Hugging Face
# Hub repository wanglab/medsam-vit-base, which is fetched automatically on first
# use. A direct Hub download is preferred; set HF_ENDPOINT to use a mirror, or
# HF_HOME to reuse a pre-populated cache on offline machines.
#
# The Google Drive copy of medsam_vit_b.pth is kept only as a fallback for users
# of the standalone `segment_anything` SAM registry. Nothing in this repository
# loads that file, and the Drive link can be rate-limited or rotated.
# ---------------------------------------------------------------------------
fetch_medsam_weights_hf() {
  python - <<'PY'
from transformers import SamModel, SamProcessor
SamModel.from_pretrained("wanglab/medsam-vit-base")
SamProcessor.from_pretrained("wanglab/medsam-vit-base")
print("MedSAM weights cached from the Hugging Face Hub.")
PY
}

fetch_medsam_weights_gdrive() {
  # Legacy Google Drive mirror of medsam_vit_b.pth (file id from the MedSAM repo).
  gdown --id 1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_ -O medsam_vit_b.pth
}

if ! fetch_medsam_weights_hf; then
  echo "Hugging Face download failed; falling back to the Google Drive mirror." >&2
  fetch_medsam_weights_gdrive || echo "gdown fallback failed; MedSAM weights are optional for the default DINO path." >&2
fi
