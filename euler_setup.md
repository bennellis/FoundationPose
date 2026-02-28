## FoundationPose conda setup (Euler)

This is the exact sequence that worked on the 3090 nodes for `foundationpose_new`.

### 1) Allocate a GPU node

```bash
srun --gpus=rtx_3090:1 --mem-per-cpu=4g --time=24:00:00 --pty --cpus-per-task=16 bash
```

If Slurm warns about `--cpus-per-task` exceeding the job allocation, it still
allocates; it is just a warning on this cluster.

### 2) Load modules

```bash
module load stack/2025-06 gcc/8.5.0 openmpi/4.1.7 boost/1.86.0 eth_proxy cuda/11.8.0
```

Boost 1.86.0 is required so `mycpp` can find `boost::system` and
`boost::program_options`.

### 3) Create and populate the conda env

```bash
cd /cluster/project/cvg/students/beellis/comind/FoundationPose
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n foundationpose_new python=3.9
conda activate foundationpose_new
conda install -y conda-forge::eigen=3.4.0
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
```

`nvdiffrast` must be installed with `--no-build-isolation` or the build fails.

### 4) Build extensions (manual fixes)

The stock `build_all_conda.sh` did not work out-of-the-box on Euler. The
manual steps below are what succeeded.

```bash
cd /cluster/project/cvg/students/beellis/comind/FoundationPose
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate foundationpose_new

# Make Boost and pybind11 discoverable for mycpp.
export Boost_INCLUDE_DIR="$BOOST_ROOT/include"
export Boost_LIBRARY_DIR="$BOOST_ROOT/lib"
export CMAKE_PREFIX_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11:$CMAKE_PREFIX_PATH"

cd mycpp
rm -rf build && mkdir -p build && cd build
cmake ..
make -j"$(nproc)"

cd ../../bundlesdf/mycuda
rm -rf build *egg* *.so

# Ensure Eigen headers are visible to nvcc.
export CPATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/eigen3:${CPATH:-}"

# Build without build isolation.
PIP_USE_PEP517=0 python -m pip install -e . --no-build-isolation --config-settings editable_mode=compat
```

### 5) Quick sanity check

```bash
python - <<'PY'
import importlib
for m in ["torch", "nvdiffrast", "common", "mycpp"]:
    importlib.import_module(m)
    print("OK:", m)
PY
```
