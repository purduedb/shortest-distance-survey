# v9-camera Job Tracker

## Training benchmark (2026-08-30) — training/csit/training

Full training block from `v9-camera-expt.md` (Surat + Workload + DIMACS), submitted with the doc's default `--account csit --partition training --qos training`. Config: `--mem 32G --job_time 04:00:00`, `--extra_python_args "--sparse_embedding"`, `--expt_name v9-camera-training`.

Results: `results/v9-camera-training/`

**Status as of 2026-08-31**: 40/40 jobs COMPLETED (exit 0:0). No errors found in any master or per-task log (checked for Traceback/CUDA OOM/Segmentation fault/Killed — none). `train_logs`=208, `saved_metrics`=208, `saved_models`=208 (16 models × 13 datasets, all accounted for); `saved_jit_models`=195 — the 13 missing are all `CatBoost_*` (one per dataset), expected since CatBoost is tree-based and isn't JIT-exported. One benign artifact: every `W_Shenzhen` model log reports `Bucket 5 ... Local MRE: nan%` — that bucket has 0.00% of test samples (empty distance bucket → 0/0), not a training failure; the headline `test_mre` metric is computed independently over the full test set and is unaffected (verified numerically).

Backup: `results/backup-v9-camera-training.tar` (uncompressed, full `results/v9-camera-training/` snapshot post-training, pre-export).

### Surat (all_pairs, time_limit=15)

| JobID    | Datasets | Models | Time Taken | Total Time | Status    |
| -------- | -------- | ------ | ---------- | ---------- | --------- |
| 11642345 | Surat    | all 16 | 03:33:51   | 04:00:00   | Completed |

### Workload (real_workload_perturb_500k, time_limit=5)

| JobID    | Datasets              | Models | Time Taken | Total Time | Status    |
| -------- | --------------------- | ------ | ---------- | ---------- | --------- |
| 11642346 | W_Jinan, W_Shenzhen   | all 16 | 02:17:21   | 04:00:00   | Completed |
| 11642347 | W_Chengdu, W_Beijing  | all 16 | 02:17:39   | 04:00:00   | Completed |
| 11642348 | W_Shanghai, W_NewYork | all 16 | 02:17:46   | 04:00:00   | Completed |
| 11642349 | W_Chicago             | all 16 | 01:09:18   | 04:00:00   | Completed |

### DIMACS (landmark_30M, n2v_backend=pyg, time_limit=60)

| JobID    | Dataset | Model group            | Time Taken | Total Time | Status    |
| -------- | ------- | ----------------------- | ---------- | ---------- | --------- |
| 11642350 | FLA     | lpnorm,landmark        | 00:04:43   | 04:00:00   | Completed |
| 11642351 | FLA     | geodnn,vdist2vec       | 02:03:48   | 04:00:00   | Completed |
| 11642352 | FLA     | ndist2vec,embeddingnn  | 02:04:37   | 04:00:00   | Completed |
| 11642353 | FLA     | gnn                    | 03:09:45   | 04:00:00   | Completed |
| 11642354 | FLA     | distancenn,aneda       | 02:04:21   | 04:00:00   | Completed |
| 11642355 | FLA     | path2vec,rne           | 02:06:53   | 04:00:00   | Completed |
| 11642356 | FLA     | catboost,catboostnn    | 02:15:55   | 04:00:00   | Completed |
| 11642357 | E       | lpnorm,landmark        | 00:06:58   | 04:00:00   | Completed |
| 11642358 | E       | geodnn,vdist2vec       | 02:05:44   | 04:00:00   | Completed |
| 11642359 | E       | ndist2vec,embeddingnn  | 02:06:36   | 04:00:00   | Completed |
| 11642360 | E       | gnn                    | 03:14:16   | 04:00:00   | Completed |
| 11642361 | E       | distancenn,aneda       | 02:04:12   | 04:00:00   | Completed |
| 11642362 | E       | path2vec,rne           | 02:08:30   | 04:00:00   | Completed |
| 11642363 | E       | catboost,catboostnn    | 02:15:20   | 04:00:00   | Completed |
| 11642364 | W       | lpnorm,landmark        | 00:08:20   | 04:00:00   | Completed |
| 11642365 | W       | geodnn,vdist2vec       | 02:05:30   | 04:00:00   | Completed |
| 11642366 | W       | ndist2vec,embeddingnn  | 02:07:24   | 04:00:00   | Completed |
| 11642367 | W       | gnn                    | 03:14:07   | 04:00:00   | Completed |
| 11642368 | W       | distancenn,aneda       | 02:06:16   | 04:00:00   | Completed |
| 11642369 | W       | path2vec,rne           | 02:04:59   | 04:00:00   | Completed |
| 11642370 | W       | catboost,catboostnn    | 02:13:07   | 04:00:00   | Completed |
| 11642371 | CTR     | lpnorm,landmark        | 00:12:11   | 04:00:00   | Completed |
| 11642372 | CTR     | geodnn,vdist2vec       | 02:05:07   | 04:00:00   | Completed |
| 11642373 | CTR     | ndist2vec,embeddingnn  | 02:06:54   | 04:00:00   | Completed |
| 11642374 | CTR     | gnn                    | 03:20:04   | 04:00:00   | Completed |
| 11642375 | CTR     | distancenn,aneda       | 02:06:26   | 04:00:00   | Completed |
| 11642376 | CTR     | path2vec,rne           | 02:09:37   | 04:00:00   | Completed |
| 11642377 | CTR     | catboost,catboostnn    | 02:15:45   | 04:00:00   | Completed |
| 11642378 | USA     | lpnorm,landmark        | 00:17:06   | 04:00:00   | Completed |
| 11642379 | USA     | geodnn,vdist2vec       | 02:06:20   | 04:00:00   | Completed |
| 11642380 | USA     | ndist2vec,embeddingnn  | 02:09:27   | 04:00:00   | Completed |
| 11642381 | USA     | gnn                    | 03:25:02   | 04:00:00   | Completed |
| 11642382 | USA     | distancenn,aneda       | 02:08:32   | 04:00:00   | Completed |
| 11642383 | USA     | path2vec,rne           | 02:11:48   | 04:00:00   | Completed |
| 11642384 | USA     | catboost,catboostnn    | 02:16:38   | 04:00:00   | Completed |

**Note**: `training` QOS on `csit` runs up to 8 concurrent jobs — the remaining 32 queue and start automatically as earlier ones finish. All 40 jobs submitted successfully and confirmed in `squeue --me`.

## ONNX Export (2026-08-31, post-training)

Not a SLURM job — ran via `srun --jobid 11648090 --overlap` on the a30 interactive session. Converted every `results/v9-camera-training/saved_jit_models/*.jit.pt` to `saved_onnx_models/*.onnx`.

**Status**: COMPLETED, 195/195 models exported, 27m14s total, no errors.

## Inference Benchmark (2026-08-31) — training/csit/training

Inference block from `v9-camera-expt.md` (TRT GPU all-13, TRT CPU 9-way split, CatBoost CPU), submitted with `--account csit --partition training --qos training --mem 32G --job_time 04:00:00`. Consumes `results/v9-camera-training/saved_onnx_models/` (TRT) and `saved_models/` (CatBoost).

Results: `results/v9-camera-training/saved_inference_metrics/{trt_cuda_100k_1m_training, trt_cpu_100k_1m_training, catboost_cpu_100k_1m_training}/`

**Status**: 11/11 jobs COMPLETED (exit 0:0). No errors in any master log (checked Traceback/Segmentation fault/Killed/error/fail/exception — none). Metrics counts match expectations: 195/195 TRT-GPU, 195/195 TRT-CPU (one per ONNX model), 13/13 CatBoost (one per dataset). SBATCH requests `--gres gpu:1` even for the CPU-device jobs — that's the script's own default node request, not something we changed.

### TRT GPU (all 13 datasets)

| JobID    | Datasets | Backend/Device | Time Taken | Total Time | Status    |
| -------- | -------- | --------------- | ---------- | ---------- | --------- |
| 11651680 | all 13   | tensorrt/cuda   | 01:48:09   | 04:00:00   | Completed |

### TRT CPU (9-way split)

| JobID    | Datasets              | Backend/Device | Time Taken | Total Time | Status    |
| -------- | --------------------- | --------------- | ---------- | ---------- | --------- |
| 11651682 | FLA                   | tensorrt/cpu    | 00:22:35   | 04:00:00   | Completed |
| 11651683 | E                     | tensorrt/cpu    | 00:29:26   | 04:00:00   | Completed |
| 11651684 | W                     | tensorrt/cpu    | 00:28:23   | 04:00:00   | Completed |
| 11651685 | CTR                   | tensorrt/cpu    | 00:49:38   | 04:00:00   | Completed |
| 11651686 | USA                   | tensorrt/cpu    | 01:47:34   | 04:00:00   | Completed |
| 11651687 | W_Shanghai, W_Beijing | tensorrt/cpu    | 00:53:53   | 04:00:00   | Completed |
| 11651688 | W_NewYork, W_Chengdu  | tensorrt/cpu    | 00:55:11   | 04:00:00   | Completed |
| 11651689 | Surat, W_Jinan        | tensorrt/cpu    | 00:41:26   | 04:00:00   | Completed |
| 11651690 | W_Chicago, W_Shenzhen | tensorrt/cpu    | 00:43:38   | 04:00:00   | Completed |

### CatBoost CPU (all 13 datasets)

| JobID    | Datasets | Backend/Device | Time Taken | Total Time | Status    |
| -------- | -------- | --------------- | ---------- | ---------- | --------- |
| 11651691 | all 13   | catboost/cpu    | 01:57:55   | 04:00:00   | Completed |
