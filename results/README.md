## Directory Structure
```
└── results/                          # Results directory containing all experiment results
    ├── README.md
    ├── <expt_name>/                      # One directory per experiment run, e.g. v9-camera-training/
    │   ├── saved_models/                     # <Model>_<Dataset>_<Query>.pt -- Model training checkpoints
    │   ├── saved_jit_models/                 # <Model>_<Dataset>_<Query>.jit.pt -- For LibTorch inference
    │   ├── saved_onnx_models/                # <Model>_<Dataset>_<Query>.onnx -- For TensorRT/ONNX Runtime inference
    │   ├── saved_metrics/                    # metrics_<Model>_<Dataset>_<Query>.json -- Training metrics
    │   ├── saved_inference_metrics/          # metrics_<Model>_<Dataset>_<Query>.json -- Inference metrics
    │   ├── train_logs/                       # train_<Model>_<Dataset>.log -- per-model training logs
    │   ├── train_jobs/                       # train-master-<jobid>.log -- SLURM master logs for the training array/job
    │   ├── inference_jobs/                   # inference-master-<jobid>.log -- SLURM master logs for model inference
    │   ├── catboost_inference_jobs/          # catboost_inference-master-<jobid>.log -- SLURM master logs for CatBoost inference
    │   └── plots/                            # PNG plots -- per-model training plots
    └── ...                               # and so on for other experiments
```
