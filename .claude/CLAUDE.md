# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Codebase for a SIGSPATIAL 2026 survey of ML-based methods (NNs, GNNs, Functional, Tree-based, baselines, etc.) for shortest-path/distance computation on road networks. Models are trained to predict shortest-path distance between node pairs, evaluated against ground-truth road-network distances.

## Environment
Always use the `myenv` conda environment (PyTorch/CUDA 12.6, PyG, TensorFlow, igraph, pecanpy, gensim, pymetis, CatBoost, sklearn, etc.) for Python commands:

```bash
conda run -n myenv python <script>.py [args]
```

- Use this for everything: training/inference/GNN/PyTorch/TensorFlow, CatBoost, node2vec embeddings (backend={openne,pyg,pecanpy}).
- Use live logs (maybe with `tee` command) so we can look at logs midway.

## SLURM
Only applicable when on Gilbreth (Purdue RCAC):
- Interactive GPU (preferred): `sinteractive -N1 -n1 -c16 --gres=gpu:1 --partition=training --mem=32G --account=csit --qos training --time 12:00:00`
- Training experimental setup: `slurm-jobs/train_expt.sh`
- Inference experimental setup: `slurm-jobs/inference_expt.sh` + `slurm-jobs/catboost_inference_expt.sh`
- Full detailed end-to-end setup for train+inference: `analysis/v9-camera-expt.md`

## Common Commands
- Run from `src/`:
```bash
# Train + evaluate a single model
python train.py --model_class rne --data_dir W_Jinan --query_dir real_workload_perturb_500k

# See train.py argparse section for full flag list (learning_rate, epochs, time_limit, seed, device, etc.)
```

- Results land in `../results/<expt_name>/` (default `../results/default`)

- Inference latency/throughput benchmark:
```bash
# All models except CatBoost -- TensorRT/ONNX (default) or LibTorch backend
python inference.py --backend tensorrt --model_path <model.onnx> --queries <queries.npz> --device cuda --batch_sizes 100000,1000000 --eval_runs 10

# CatBoost -- CPU-only
python catboost_inference.py --model_path <CatBoost_model.pt> --queries <queries.npz> --batch_sizes 100000,1000000 --eval_runs 10
```

## Directory Layout

**Data flow:** raw network → `.edges`/`.nodes` → `.queries` → `train.py` (loads graph, trains `BaseModel` subclass) → `.pt` + `.jit.pt` checkpoints → MAE/MRE/latency eval.

**`src/`**
- **`src/train.py`** - single entry point, argparse-driven. Dispatches to ~13 model classes via `--model_class` (`if/elif`, no registry). Loads graph via `igraph`, calls `model.fit(...)`, saves plain + JIT checkpoints, evaluates and saves metrics/plots.

- **`src/inference.py`** - latency/throughput/accuracy benchmark for all non-CatBoost models; `--backend tensorrt` (default, ONNX Runtime) or `libtorch`, loads `.onnx`/`.jit.pt`, samples GPU stats during eval.

- **`src/catboost_inference.py`** - same benchmark for CatBoost, CPU-only (`.pt` via CatBoost's own `predict()`).

- **`src/models/basemodel.py`** - `BaseModel(nn.Module)`: shared `fit`/`_train_step`/`evaluate`. Subclasses usually just implement `forward(x1, x2)`; non-gradient models (`landmark.py`, `lpnorm.py`) override `fit`/`evaluate`.

- **`src/models/`** - one file per family (default config in `train_expt.sh`):
  - Baselines (non-learned): `lpnorm.py` - paper alias Manhattan (`--p_norm=1`) or Euclidean (`--p_norm=2`); `landmark.py` - paper alias Landmark_rn/Landmark_km via `--landmark_selection=random/kmeans` and `--select_landmarks_from_train`
  - NNs: `geodnn.py`, `distancenn.py`, `embeddingnn.py`, `ndist2vec.py`, `vdist2vec.py`, `catboostnn.py` (not a catboost model, aliased as `LandmarkNN` in paper)
  - GNNs: `rgnndist2vec.py` - `--gnn_layer`: gcn/sage/gat/etc.
  - Functional: `path2vec.py`, `aneda.py`, `rne.py`
  - Tree-based: `catboostmodel.py`
  - Misc: `sparse_matrix_model` - no learning, pure lookup, run standalone for QT/model-size benchmarks.

- **`src/utils/`** - `data_utils.py` (graph/query/embedding I/O), `torch_utils.py` (device/optimizer/save-load incl. JIT), `plot_utils.py`.

- **`src/data_preprocess/`** - offline scripts (schema of each output in `data/<name>/` below):
  - `download_and_preprocess_data.py` → `data/<data_dir>/{*.edges, *.nodes}`
  - `generate_query_data.py` → `data/<data_name>/<query_dir>/<data_name>_*.queries.npz`
  - `generate_parts_file_rne.py` → `data/<data_name>/<data_name>.parts.npz`
  - `generate_landmark_distances.py` → `data/<data_dir>/landmark_dim<N>.embeddings.npz`
  - `generate_node2vec_embeds_{pyg,pecanpy}.py` → `data/<data_dir>/node2vec_dim<N>_epochs<N>_unweighted_{pyg,pecanpy}.embeddings.npz`
  - `generate_dataset_stats.py` → `data/<data_name>/stats.json`
  - `export_onnx_models.py` → converts `saved_jit_models/*.jit.pt` to `saved_onnx_models/*.onnx` (required for TensorRT inference)

**`data/`**
- **`data/<name>/`** - 1-indexed node IDs; full spec in `data/README.md`:
  - `<name>.edges` - src,dst,weight in meters, comma-separated no header
  - `<name>.nodes` - id,x,y - projected coords (EPSG:5070 for DIMACS sets, per-city UTM for workload/`Surat` datasets), comma-separated no header
  - `*.parts.npz` - RNE hierarchy, key `data`, int64, shape `(n_nodes, n_levels)`
  - `node2vec_*.embeddings.npz` - key `data`, float64 - hstack of int node_id col0 upcasts the float32 embedding, shape `(n_nodes, 1+N)`
  - `landmark_*.embeddings.npz` - key `data`, float64, shape `(n_nodes, 1+N)` col0=node_id; key `comment`
  - query dirs with `*.queries.npz` - keys: `src`,`dst` int64, `dist` float32
  - `W_*` datasets - workload-driven (real trajectories: Jinan, Shenzhen, Chengdu, Beijing, Shanghai, NewYork, Chicago); other datasets are DIMACS benchmarks (`FLA`,`E`,`W`,`CTR`,`USA`) or figshare (`Surat`), queried via synthetically generated query workloads.

- **`results/<expt_name>/`** (full schema in `results/README.md`)
  - Training metrics: `saved_metrics/metrics_<Model>_<Dataset>_<Query>.json`
  - Inference metrics: `saved_inference_metrics/metrics_<Model>_<Dataset>_<Query>.json`
  - Other artifacts: `plots/*.png`, `saved_models/*.pt`, `saved_jit_models/*.jit.pt`, `saved_onnx_models/*.onnx`
  - SLURM master logs: `*_jobs/*-master-<jobid>.log`
  - SLURM (train) worker logs in `train_logs/train_<model>_<dataset>.log`
  - Latest experimental results: `results/v9-camera-training/`

- **`slurm-jobs/`** - SLURM job scripts: `train_expt.sh` (training), `inference_expt.sh` + `catboost_inference_expt.sh` (inference).

- **`claude_work/`** - scratch space for Claude sessions (gitignored).

- **`non_ml_index/Hierarchical-Cut-Labelling/`** - C++ Hierarchical Cut Labelling index, ground-truth oracle for query workloads (`make build DATA_NAME=<name>`, or `bash run_all.sh` for all datasets).

- **`archive/`** - vendored reference repos (`OpenNE`, `ndist2vec`, `vdist2vec`, etc.).

- **`analysis/`** - downstream consumer of `results/<expt_name>/`; full end-to-end command sequence (preprocessing → training → ONNX export → inference → extraction) is in `analysis/v9-camera-expt.md`.
  - `extract_metrics.py` → `metrics_data.json` (raw metrics)
  - `tables/build_tables.py` → `tables/all_tables.txt` (markdown table per metric)
  - `dashboard/export_dashboard_data.py` → `dashboard/dashboard.html` (dashboard for visualizing metrics)
  - `plots/fig*.py` → `plots/*.pdf` (figures used in paper)

## Known gotchas
- Pandas: use `.str.split(" ").str[0]`, not `.astype(str).map(lambda x: x.split(" ")[0])` (breaks on NaN).
- macOS: `OMP: Error #15: Initializing libomp.dylib...` (duplicate libomp.dylib) from conda + torch/networkit both bundling OpenMP - set `KMP_DUPLICATE_LIB_OK=TRUE` to work around it.

## Instructions
- Commit messages follow the `/staged-commit` format (see global instructions); no co-authored-by lines.
- **Don't be too verbose in answers. Be very concise. Be terse.**
