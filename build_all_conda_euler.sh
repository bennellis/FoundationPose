#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Load modules if available.
if command -v module >/dev/null 2>&1; then
  module load stack/2025-06 gcc/8.5.0 openmpi/4.1.7 boost/1.86.0 eth_proxy cuda/11.8.0
fi

# Activate conda in non-interactive shells.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate foundationpose_new

# Prefer conda libs/includes.
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"
export Boost_INCLUDE_DIR="${BOOST_ROOT}/include"
export Boost_LIBRARY_DIR="${BOOST_ROOT}/lib"
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}/lib/python3.9/site-packages/pybind11/share/cmake/pybind11:${CMAKE_PREFIX_PATH}"

# Optional but often needed for CUDA extensions when no GPU is visible at build time.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# Ensure Eigen headers are visible to nvcc.
export CPATH="${CONDA_PREFIX}/include:${CONDA_PREFIX}/include/eigen3:${CPATH:-}"

# Install mycpp.
cd "${PROJ_ROOT}/mycpp/"
rm -rf build && mkdir -p build && cd build
cmake ..
make -j"$(nproc)"

# Install mycuda without build isolation.
cd "${PROJ_ROOT}/bundlesdf/mycuda"
rm -rf build *egg* *.so
PIP_USE_PEP517=0 python -m pip install -e . --no-build-isolation --config-settings editable_mode=compat

cd "${PROJ_ROOT}"
