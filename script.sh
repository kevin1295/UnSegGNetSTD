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
# Note: nothing in this repository loads medsam_vit_b.pth. `features_extract.py`
# only imports `sam_model_registry` at module level and never builds a SAM model,
# so MedSAM weights are optional; the previous Google Drive (gdown) fallback was
# removed because the link is dead and gdown is not a requirement.
# ---------------------------------------------------------------------------
fetch_medsam_weights_hf() {
  python - <<'PY'
from transformers import SamModel, SamProcessor
SamModel.from_pretrained("wanglab/medsam-vit-base")
SamProcessor.from_pretrained("wanglab/medsam-vit-base")
print("MedSAM weights cached from the Hugging Face Hub.")
PY
}

if ! fetch_medsam_weights_hf; then
  echo "Hugging Face download failed." >&2
  echo "MedSAM weights are optional: the default DINO pipeline does not read them," >&2
  echo "and features_extract.py only imports segment_anything without building a model." >&2
fi
