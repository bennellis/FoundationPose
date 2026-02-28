#!/usr/bin/env bash
set -euo pipefail

# Load CUDA if module system is available.
if command -v module >/dev/null 2>&1; then
  module load cuda
fi

# Activate conda in non-interactive shells.
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create and activate the new environment.
conda create -y -n foundationpose_new python=3.9
conda activate foundationpose_new

# Install Eigen3 3.4.0 under conda environment.
conda install -y conda-forge::eigen=3.4.0

# Install dependencies.
python -m pip install -r requirements.txt

# Install NVDiffRast.
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (optional, needed if running model-free setup).
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D.
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions.
CMAKE_PREFIX_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11" bash build_all_conda.sh
