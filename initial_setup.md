# Initial Setup

---

# PyTorch and PyTorch Geometric Setup on Gilbreth HPC Cluster

This guide provides instructions for setting up a conda environment with PyTorch and PyTorch Geometric on the Gilbreth HPC cluster at Purdue University. If you are installing on a local system, you can skip the `ssh` and `module` commands, but the rest of the instructions should still apply.

## Environment Setup

Skip to the **Environment Usage** section if you already have a conda environment set up with PyTorch and PyTorch Geometric.

### Prerequisites

- As of March 2025, CUDA 12.1 and CUDA 12.6 are available on Gilbreth
- We'll install PyTorch and PyTorch Geometric with CUDA 12.1 (as CUDA 12.6 is backwards compatible)
- Note: The `~/.conda` and `~/.cache` directories can be safely removed for a fresh start, if needed


### Login to Gilbreth
Login to the Gilbreth cluster:

```bash
# Option 1: SSH into the cluster
ssh username@gilbreth.rcac.purdue.edu

# Option 2: SSH into a specific, say `fe00`, front-end node (for using `tmux`)
ssh username@gilbreth-fe00.rcac.purdue.edu
```

> **Note**: Use `-fe00` to connect to the same front-end node as `tmux` sessions are running on, else you may not be able to access them. If you are not using `tmux`, you can connect to any front-end node.


### Create Conda Environment
First, check your currently loaded modules:

```bash
module list
```

Load the conda module (note: CUDA/12.6 should already be loaded):

```bash
module load conda
```

One way to automatically create conda environments is to use the `environment.yml` file. This file contains a list of packages and their versions that are required for your project. To create a new conda environment using this file (and skip installing packages one by one), you can use the following command:

```bash
# This will create environment (named `myenv`) in the home directory
conda env create -f environment.yml
```

Another way is to create a new conda environment (named `myenv`) manually with the following command:

```bash
# Option 1 (Recommended): Create environment in home directory
conda create -n myenv python=3.10 ipython ipykernel -y

# Option 2: Create environment in scratch directory (if low on space in home directory)
conda create -p ~/scratch/copy-myenv python=3.10 ipython ipykernel -y
```

> **Note**: If you encounter connection errors like "Network is unreachable" when collecting package metadata, simply retry the command.

Activate the environment:

```bash
# Option 1 (Recommended): For home directory installation
conda activate myenv

# Option 2: For scratch directory installation
conda activate ~/scratch/copy-myenv
```

Optionally, if you want to use this environment in a Jupyter Notebook, register the kernel:

```bash
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
# Output:
# Installed kernelspec myenv in /home/username/.local/share/jupyter/kernels/myenv
```


### Installing PyTorch
Install [PyTorch](https://pytorch.org/get-started/previous-versions/) with CUDA 12.1 support:

```bash
# CUDA Support (Linux and Windows)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# MPS Support (MacOS)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch -y

# CPU Only (Linux and Windows)
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 cpuonly -c pytorch -y
```

Verify the installation:

```bash
# CUDA Support (Linux and Windows)
python -c "import torch; print('Torch Version: ', torch.__version__); print('Torch CUDA Version: ', torch.version.cuda); print('CUDA available: ', torch.cuda.is_available()); print('CUDA device: ', torch.cuda.current_device()); print('CUDA device name: ', torch.cuda.get_device_name())"

# MPS Support (MacOS)
python -c "import torch; print('Torch Version: ', torch.__version__); print('MPS available: ', torch.backends.mps.is_available())"

# CPU version (Linux and Windows)
python -c "import torch; print('Torch Version: ', torch.__version__)"
```

> **Note**: If you don't have a GPU, you may install the CPU version of PyTorch and it will work fine (just a bit slower).


### Installing PyTorch Geometric
Install the core [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) package (same for CUDA/MPS/cpu versions):

```bash
pip install torch_geometric==2.6.1
```

Verify the PyG installation:

```bash
python -c "import torch_geometric; print(f'PyTorch Geometric: {torch_geometric.__version__}')"
```


### Installing TensorFlow
Install [TensorFlow](https://www.tensorflow.org/install/pip) with CUDA support:

```bash
# CUDA Support (Linux)
pip install tensorflow[and-cuda]==2.19.0
```
Verify the TensorFlow installation:

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')"
```


### Additional Packages
Install commonly used data science packages:

```bash
conda install -c anaconda numpy pandas seaborn matplotlib networkx tqdm scipy -y
```

Verify the installation:

```bash
python -c "import pandas, numpy, networkx, seaborn, tqdm, matplotlib; print('Done')"
```


### Cleanup
Clean up the conda cache to save space:

```bash
conda clean --all
```

Reset modules to default:

```bash
module load rcac
```

Clean up the pip cache (optional):

```bash
pip cache purge
```

> **Note**: The pip cache is present in `~/.cache/pip` and can be safely removed if you want to free up space. To check the size of the pip cache, you can use the command `du -sh ~/.cache/pip`.

## Environment Usage
To use the environment, load the conda module, activate the environment, and run your scripts:

```bash
# Option 1 (Recommended): If created in home directory
module load conda
conda activate myenv

# Option 2: If created in scratch directory
module load conda
conda activate ~/scratch/copy-myenv
```

## Notes
- I found that it's better to create the conda environment in the home directory rather than the scratch directory, as while the latter is on `lustre`, it is slower in loading a lot of small files (e.g., while importing torch in python) compared to the home directory which is on `nfs`. Lustre is better suited for handling large files (not a lot of small files). Additionally, the scratch directory is not backed up, so if you lose your files, you may not be able to recover them. I experimented timing the import of torch in both directories and found that it takes around 7s in the home directory and around 13s in the scratch directory.
- No need to explicitly load `module load cuda/12.1` as the current CUDA/12.6 is backwards compatible
