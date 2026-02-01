# Project Overview
This project focuses on preprocessing road network graphs and creating datasets for machine learning models. The `src` directory contains the implementations, and the `data` directory holds the preprocessed datasets.


## Directory Structure
```
├── .gitignore
├── environment.yml         # The requirements file for reproducing the environment
├── initial_setup.md        # Instructions for setting up the environment
├── data/                   # Preprocessed datasets for various road networks
├── non_ml_index/           # Codebase for non-ML index evaluation
├── slurm-jobs/             # SLURM job scripts for running experiments on HPC clusters
├── results/                # Saved logs, plots, and model checkpoints
├── third_party/            # Third-party repositories or libraries
├── src/                    # Source code for this project
├── LICENSE
└── README.md
```

## Requirements
Our code has been developed using PyTorch, PyTorch Geometric and Tensorflow in Python 3.10. Please refer to `environment.yml` for the complete list of dependencies and `initial_setup.md` for detailed instructions on setting up the environment.

```bash
# This will create environment (named `myenv`)
conda env create -f environment.yml
```

## Usage
0. Git clone the repository:
    ```bash
    # Clone the repository to a specific directory
    git clone https://github.com/purduedb/shortest-distance-survey

    # Change to the project directory
    cd shortest-distance-survey
    ```

1. Activate the environment (follow instructions in `initial_setup.md`):
    ```bash
    # Load the conda module if using an HPC cluster, else skip `module` commands
    module load conda

    # Activate the conda environment
    conda activate myenv
    ```

2. Use sample preprocessed datasets: `W_Jinan` or `Surat_subgraph` available in `data/` directory. Other workload-driven datasets are also available for download, [here](https://purdue0-my.sharepoint.com/:f:/g/personal/gchoudha_purdue_edu/IgCJZ35eEvaHRY18UCM74rxxAQVi_wEO-2d34VRndJPIUvU?e=VTigxg): `W_Shenzhen`, `W_Chengdu`, `W_Beijing`, `W_Shanghai`, `W_NewYork`, `W_Chicago`. Refer to `data/README.md` for more details on the datasets.

4. Train and evaluate a model:
    ```bash
    # Change to source directory
    cd ~/src

    # Run RNE model with Jinan dataset
    python train.py --model_class rne --data_dir W_Jinan --query_dir real_workload_perturb_500k
    ```

    NOTE: Additional parameters, e.g., time_limit, learning_rate, seed, etc. may also be specified. Refer to argparse section in `train.py` for full list. Refer to `slurm-jobs/urban_expt.sh` for other model configurations.

5. Optionally, train multiple models using SLURM scripts:
    ```bash
    # Change to directory containing scripts
    cd ~/slurm-jobs

    ## Run training script (modify configuration in the script as needed)
    # (Optional) Dry run
    bash urban_expt.sh

    # (Optional) Execute on local
    bash urban_expt.sh --execute

    # Execute through SLURM
    bash urban_expt.sh --execute --slurm
    ```


## FAQ
...

## Acknowledgements
...

## Citation
If you find this code useful, please cite the following:

```
...
```

## Contact
```
Gautam Choudhary (PhD Student, Purdue University)
Email: gchoudha@purdue.edu
```

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
