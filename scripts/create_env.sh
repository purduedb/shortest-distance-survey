#!/usr/bin/env bash
# Usage: bash scripts/create_env.sh
set -euo pipefail

# Set the repository root and environment file paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/environment.yml"
ENV_NAME="$(grep '^name:' "$ENV_FILE" | awk '{print $2}')"

# Load conda module (HPC systems with Lmod/Environment Modules only)
if command -v module &>/dev/null; then
    module load conda
fi

# Remove the existing environment, if it exists
conda env remove -n "$ENV_NAME" -y || true

# Create the new environment
conda env create -f "$ENV_FILE"

# Clean up conda cache
conda clean --all -y
conda run -n "$ENV_NAME" pip cache purge

# Verify the installation
conda run -n "$ENV_NAME" python -c "
import numpy, pandas
import sklearn, catboost, gensim, scipy
import igraph, networkit, networkx
import torch, torch_geometric
import tensorflow as tf
import onnx, onnxruntime as ort
print('torch          :', torch.__version__)
print('pyg            :', torch_geometric.__version__)
print('tensorflow     :', tf.__version__)
print('scipy          :', scipy.__version__)
print('numpy          :', numpy.__version__)
print('pandas         :', pandas.__version__)
print('sklearn        :', sklearn.__version__)
print('catboost       :', catboost.__version__)
print('gensim         :', gensim.__version__)
print('networkx       :', networkx.__version__)
print('igraph         :', igraph.__version__)
print('networkit      :', networkit.__version__)
print()
print('CUDA available :', torch.cuda.is_available())
print('CUDA version   :', torch.version.cuda)
print('GPU count      :', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  - GPU {i}      : {torch.cuda.get_device_name(i)}')
print()
print('onnx           :', onnx.__version__)
print('onnxruntime    :', ort.__version__)
print('ORT providers  :', ort.get_available_providers())
"

echo "Done."

# KNOWN ISSUE: conda-forge's scipy/igraph builds need a newer libstdc++ ABI than the
# system one, so `import scipy`/`import igraph` fail with "CXXABI_1.3.15 not found"
# unless the env's own lib/ is found first. Two fixes:
#   Manual:  export LD_LIBRARY_PATH="${CONDA_ENVS_PATH:-$HOME/.conda/envs}/$ENV_NAME/lib:$LD_LIBRARY_PATH"
#   Hook:    drop these into the env so it's automatic on `conda activate`/`deactivate`;
#            must be re-added after every `create_env.sh` rebuild.
#              etc/conda/activate.d/libstdcxx.sh:
#                export _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
#                export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
#              etc/conda/deactivate.d/libstdcxx.sh:
#                export LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
#                unset _OLD_LD_LIBRARY_PATH
