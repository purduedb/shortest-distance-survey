# v9-camera Experiment Plan

## Commands

### Data Preprocessing

Assumes raw `.edges`/`.nodes`/`raw_workload` already downloaded (`download_and_preprocess_data.py`). To be run in an sinteractive session (same config as SLURM scripts):

```bash
cd ~/scratch/shortest-distance-survey/src/data_preprocess

# 1. Generate RNE `.parts`
bash generate_parts_file_rne.sh               # -> data/<name>/<name>.parts.npz

# 2. Generate node2vec embeddings
## 2.1 OpenNE embeddings -- consumed by Surat + W_* training (n2v_backend=openne)
bash generate_node2vec_embeds_openne.sh       # -> data/<name>/node2vec_dim64_epochs1_unweighted_openne.embeddings.npz

## 2.2 PyG embeddings -- consumed by DIMACS training (n2v_backend=pyg)
bash generate_node2vec_embeds_pyg.sh          # -> data/<name>/node2vec_dim64_epochs1_unweighted_pyg.embeddings.npz

# 3. Generate landmark distance embeddings for CatBoost and CatBoostNN (aka. LandmarkNN)
bash generate_landmark_distances.sh           # -> data/<name>/landmark_dim61.embeddings.npz

# 4. Generate query workloads
## 4.1 First build `.hl` index binaries for ground truth
cd ../../non_ml_index/Hierarchical-Cut-Labelling
make compile
bash run_all.sh                               # -> saved_indexes/<name>.hl
cd ../../src/data_preprocess

## 4.2 Generate query workloads (All-pairs - Surat, Workload-driven - W_*, Landmark-based - DIMACS)
bash generate_query_data.sh                   # -> data/<name>/<query_dir>/<name>_{train,val,test}.queries.npz
```

### Training Benchmark

#### [Info] Training Jobs actual runtime vs `--time_limit` (from v8-camera-training logs)

| `--time_limit`   | n tasks | min      | max       | mean      | median    |
|-----------------:|--------:|---------:|----------:|----------:|----------:|
| 5 min (Workload) | 112     | 0.16 min | 5.51 min  | 4.30 min  | 5.22 min  |
| 15 min (Surat)   | 16      | 0.70 min | 17.33 min | 13.37 min | 15.97 min |
| 60 min (DIMACS)  | 80      | 1.43 min | 73.37 min | 52.86 min | 62.96 min |

#### [Info] Training Job groups

| # | Group (Datasets)         | query_dir                    | time_limit | job_time | n2v_backend | models   | sparse flag |
|--:|-------------------------:|-----------------------------:|-----------:|---------:|------------:|---------:|------------:|
| 1 | Surat                    | `all_pairs`                  |     15 min |  4h each | openne      | all 16   | yes         |
| 2 | Worklad (W_* datasets)   | `real_workload_perturb_500k` |      5 min |  4h each | openne      | all 16   | yes         |
| 3 | DIMACS (FLA/E/W/CTR/USA) | `landmark_30M`               |     60 min |  4h each | **pyg**     | all 16   | yes         |

Three `train_expt.sh` submission rounds, one per group above. Each round splits into multiple `sbatch` calls to stay under the 4h job_time while covering all 16 models per dataset.

```bash
PARTITION=training
ACCOUNT=csit
QOS=training

cd ~/scratch/shortest-distance-survey/slurm-jobs

# Sparse - Surat - All pairs - 15 min               -- 3h34m (actual)
bash train_expt.sh \
    --expt_name v9-camera-${PARTITION} \
    --filter_datasets Surat \
    --query_dir all_pairs \
    --time_limit 15 \
    --extra_python_args "--sparse_embedding" \
    --partition $PARTITION \
    --account $ACCOUNT \
    --qos $QOS \
    --mem 32G \
    --job_time 04:00:00 \
    -e -s

# Sparse - W_* - workload perturbed - 5 min         -- (actual)
    # W_Jinan,W_Shenzhen                            -- 2h17m
    # W_Chengdu,W_Beijing                           -- 2h18m
    # W_Shanghai,W_NewYork                          -- 2h18m
    # W_Chicago                                     -- 1h09m
for FILTER_DATASETS in \
    "W_Jinan,W_Shenzhen" \
    "W_Chengdu,W_Beijing" \
    "W_Shanghai,W_NewYork" \
    "W_Chicago" \
; do
  bash train_expt.sh \
      --expt_name v9-camera-${PARTITION} \
      --filter_datasets "$FILTER_DATASETS" \
      --query_dir real_workload_perturb_500k \
      --time_limit 5 \
      --extra_python_args "--sparse_embedding" \
      --partition $PARTITION \
      --account $ACCOUNT \
      --qos $QOS \
      --mem 32G \
      --job_time 04:00:00 \
      -e -s
done

# Sparse - FLA/E/W/CTR/USA - landmark_30M - 60 min  -- (actual range across 5 datasets)
for ds in FLA E W CTR USA; do
  for model_group in \
      "lpnorm,landmark" \
      "geodnn,vdist2vec" \
      "ndist2vec,embeddingnn" \
      "gnn" \
      "distancenn,aneda" \
      "path2vec,rne" \
      "catboost,catboostnn" \
  ; do
    #   lpnorm,landmark                             -- 5-17m
    #   geodnn,vdist2vec                            -- 2h04-06m
    #   ndist2vec,embeddingnn                       -- 2h05-09m
    #   gnn                                         -- 3h10-25m
    #   distancenn,aneda                            -- 2h04-09m
    #   path2vec,rne                                -- 2h05-12m
    #   catboost,catboostnn                         -- 2h13-17m
    bash train_expt.sh \
        --expt_name v9-camera-${PARTITION} \
        --filter_datasets $ds \
        --filter_models "$model_group" \
        --query_dir landmark_30M \
        --n2v_backend pyg \
        --time_limit 60 \
        --extra_python_args "--sparse_embedding" \
        --partition $PARTITION \
        --account $ACCOUNT \
        --qos $QOS \
        --mem 32G \
        --job_time 04:00:00 \
        -e -s
  done
done
```

### ONNX Export `.pt` Models

Converts every `saved_jit_models/*.jit.pt` to `saved_onnx_models/*.onnx`, required by the TensorRT inference benchmark below. Not a SLURM job — run directly after training completes (skips files already exported).

```bash
# Export *.jit.pt --> *.onnx                        -- 27m14s
PARTITION=training

cd ~/scratch/shortest-distance-survey
bash src/data_preprocess/export_onnx_models.sh results/v9-camera-${PARTITION}
```

### Inference Benchmark (bs=100k/1M config)

Batch size 1M beats smaller sizes on latency and throughput (clearly on GPU, mostly on CPU) — carried over from v8's findings. Separate `*_100k_1m` output dirs keep this from overwriting the earlier bs=1000/10000 runs.

```bash
PARTITION=training
ACCOUNT=csit
QOS=training

cd ~/scratch/shortest-distance-survey/slurm-jobs

# TRT GPU — all 13 datasets                         -- 1h48m (actual)
bash inference_expt.sh \
    --model_dir ../results/v9-camera-${PARTITION} \
    --backend tensorrt --device cuda \
    --n_queries 10000000 \
    --batch_sizes 100000,1000000 \
    --output_dir ../results/v9-camera-${PARTITION}/saved_inference_metrics/trt_cuda_100k_1m_${PARTITION} \
    --partition $PARTITION \
    --account $ACCOUNT \
    --qos $QOS \
    --mem 32G \
    --job_time 04:00:00 \
    -s -e

# TRT CPU - 9 way split                             -- (actual)
    # FLA                                           -- 23m
    # E                                             -- 29m
    # W                                             -- 28m
    # CTR                                           -- 50m
    # USA                                           -- 1h48m
    # W_Shanghai,W_Beijing                          -- 54m
    # W_NewYork,W_Chengdu                           -- 55m
    # Surat,W_Jinan                                 -- 41m
    # W_Chicago,W_Shenzhen                          -- 44m
for FILTER_DATASETS in \
    "FLA" \
    "E" \
    "W" \
    "CTR" \
    "USA" \
    "W_Shanghai,W_Beijing" \
    "W_NewYork,W_Chengdu" \
    "Surat,W_Jinan" \
    "W_Chicago,W_Shenzhen" \
; do
  bash inference_expt.sh \
      --model_dir ../results/v9-camera-${PARTITION} \
      --backend tensorrt --device cpu \
      --filter_datasets "$FILTER_DATASETS" \
      --n_queries 10000000 \
      --batch_sizes 100000,1000000 \
      --output_dir ../results/v9-camera-${PARTITION}/saved_inference_metrics/trt_cpu_100k_1m_${PARTITION} \
      --partition $PARTITION \
      --account $ACCOUNT \
      --qos $QOS \
      --mem 32G \
      --job_time 04:00:00 \
      -s -e
done

# Catboost CPU                                      -- 1h58m (actual)
bash catboost_inference_expt.sh \
    --saved_models_dir ../results/v9-camera-${PARTITION}/saved_models \
    --n_queries 10000000 \
    --batch_sizes 100000,1000000 \
    --output_dir ../results/v9-camera-${PARTITION}/saved_inference_metrics/catboost_cpu_100k_1m_${PARTITION} \
    --partition $PARTITION \
    --account $ACCOUNT \
    --qos $QOS \
    --mem 32G \
    --job_time 04:00:00 \
    -s -e
```

### Batch-size sweep ablation (USA only, bs=100,1k,...,1M)

```bash
PARTITION=training
ACCOUNT=csit
QOS=training
BS_SWEEP="100,1000,10000,100000,1000000"

cd ~/scratch/shortest-distance-survey/slurm-jobs

# TRT GPU -- USA only                                   -- 18m35s
bash inference_expt.sh \
    --model_dir ../results/v9-camera-${PARTITION} \
    --backend tensorrt --device cuda \
    --filter_datasets "USA" \
    --n_queries 10000000 \
    --batch_sizes $BS_SWEEP \
    --output_dir ../results/v9-camera-${PARTITION}/saved_inference_metrics/bs_sweep_${PARTITION}/trt_cuda \
    --partition $PARTITION \
    --account $ACCOUNT \
    --qos $QOS \
    --mem 32G \
    --job_time 04:00:00 \
    -s -e

# TRT CPU -- USA only, 5 model groups                   -- (actual)
    # G1: Manhattan, Landmark_rn, Landmark_km, GeoDNN   -- 1h08m
    # G2: Vdist2vec, Ndist2vec, CatBoostNN              -- 2h53m
    # G3: SAGE, GAT, GCN                                -- 2h50m
    # G4: EmbeddingNN_mean, DistanceNN_sub              -- 2h00m
    # G5: ANEDA, Path2vec, RNE                          -- 2h48m
for MODELS_GROUP in \
    "Manhattan,Landmark_random_subset,Landmark_kmeans_subset,GeoDNN" \
    "Vdist2vec,Ndist2vec,CatBoostNN" \
    "SAGE,GAT,GCN" \
    "EmbeddingNN_mean,DistanceNN_sub" \
    "ANEDA,Path2vec,RNE" \
; do
  bash inference_expt.sh \
      --model_dir ../results/v9-camera-${PARTITION} \
      --backend tensorrt --device cpu \
      --filter_datasets "USA" \
      --filter_models "$MODELS_GROUP" \
      --n_queries 10000000 \
      --batch_sizes $BS_SWEEP \
      --output_dir ../results/v9-camera-${PARTITION}/saved_inference_metrics/bs_sweep_${PARTITION}/trt_cpu \
      --partition $PARTITION \
      --account $ACCOUNT \
      --qos $QOS \
      --mem 32G \
      --job_time 04:00:00 \
      -s -e
done

# CatBoost CPU -- USA only                              -- 20m29s
bash catboost_inference_expt.sh \
    --saved_models_dir ../results/v9-camera-${PARTITION}/saved_models \
    --filter_datasets "USA" \
    --n_queries 10000000 \
    --batch_sizes $BS_SWEEP \
    --output_dir ../results/v9-camera-${PARTITION}/saved_inference_metrics/bs_sweep_${PARTITION}/catboost_cpu \
    --partition $PARTITION \
    --account $ACCOUNT \
    --qos $QOS \
    --mem 32G \
    --job_time 04:00:00 \
    -s -e
```

## Extract results (downstream consumer)

```bash
cd ~/scratch/shortest-distance-survey/analysis

# Raw Metrics (metrics_data.json)
python extract_metrics.py

# Tables (tables/all_tables.txt)
cd tables && python build_tables.py && cd ..

# Dashboard (dashboard/dashboard.html)
cd dashboard && python export_dashboard_data.py && cd ..

# Plots (plots/*.pdf)
cd plots
python fig1_performance.py
# ... and so on
```
