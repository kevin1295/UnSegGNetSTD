#!/bin/bash

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
pip install git+https://github.com/bowang-lab/MedSAM.git

model_id="1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_&confirm=t"
gdown $model_id